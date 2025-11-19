use crate::BodyParseBehavior;
use crate::LocalTransform;
use crate::NameValuePair;
use crate::TransformationError;
use crate::TransformationOps;
use anyhow::{Context, Error, Result};
use base64::{
    engine::general_purpose::{STANDARD, STANDARD_NO_PAD, URL_SAFE},
    Engine,
};
use minijinja::{Environment, State};
use once_cell::sync::Lazy;
use rand::Rng;
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::env;

const BODY: &str = "body";
const CONTEXT: &str = "context";

static ENV: Lazy<Environment<'static>> = Lazy::new(new_jinja_env);

static GLOBALS_LOOKUP: Lazy<HashSet<&'static str>> =
    Lazy::new(|| ENV.globals().map(|(k, _)| k).collect());

// substring can be called with either two or three arguments --
// the first argument is the string to be modified, the second is the start position
// of the substring, and the optional third argument is the length of the substring.
// If the third argument is not provided or invalid, the substring will extend to
// the end of the string.
fn substring(input: &str, start: usize, len: Option<usize>) -> String {
    let input_len = input.len();
    if start >= input_len {
        return String::default();
    }

    let mut end = input_len;
    if let Some(len) = len {
        if start + len <= input_len {
            end = start + len
        }
    }

    input[start..end].to_string()
}

fn header(state: &State, key: &str) -> String {
    let headers = state.lookup("headers");
    let Some(headers) = headers else {
        return String::default();
    };

    let Some(header_map) = <HashMap<String, String>>::deserialize(headers.clone()).ok() else {
        return String::default();
    };

    header_map.get(key).cloned().unwrap_or_default()
}

fn request_header(state: &State, key: &str) -> String {
    let headers = state.lookup("request_headers");
    let Some(headers) = headers else {
        return String::default();
    };

    let Some(header_map) = <HashMap<String, String>>::deserialize(headers.clone()).ok() else {
        return String::default();
    };
    header_map.get(key).cloned().unwrap_or_default()
}

fn trim_outer_quotes(s: &str) -> &str {
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

fn raw_string(value: &str) -> String {
    // Not sure if this is exactly the correct behavior for this function. In the C++ version,
    // the native json object can be added to the context directly and that json object can dump
    // out the raw string without un-escaping. Here, it's several layers of deserializing and serializing
    // from serde_json::from_slice() -> constructing a BTreeMap -> adding that to the context.
    // There is no way to get back the original raw_string. So, escaping the string again is the closest I
    // can get. After escaping, the resulting string has extra double quote around the original string, so
    // need to trim them (somehow the need for trimming the double quotes is exactly the same in the C++
    // code)
    match serde_json::to_string(value) {
        Ok(s) => trim_outer_quotes(&s).to_string(),
        Err(_) => String::default(),
    }
}

fn base64_encode(input: &[u8]) -> String {
    STANDARD.encode(input)
}

fn base64_decode(input: &str) -> String {
    STANDARD
        .decode(input)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_default()
}

fn base64url_encode(input: &[u8]) -> String {
    URL_SAFE.encode(input)
}

fn base64url_decode(input: &str) -> String {
    URL_SAFE
        .decode(input)
        .ok()
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_default()
}

fn get_env(env_var: &str) -> String {
    env::var(env_var).unwrap_or_default()
}

fn replace_with_random(input: &str, to_replace: &str) -> String {
    // TODO: in the C++ version, the pattern is generated once per "to_replace" string
    //       and get re-used for all calls within the request context but I cannot find
    //       a way to do this here yet
    let mut rng = rand::rng();
    let high: u64 = rng.random();
    let low: u64 = rng.random();
    let mut random = [0u8; 16];
    random[..8].copy_from_slice(&low.to_le_bytes());
    random[8..].copy_from_slice(&high.to_le_bytes());

    let pattern = STANDARD_NO_PAD.encode(random);
    input.replace(to_replace, &pattern)
}

fn body(state: &State) -> String {
    println!("body() called");
    state.lookup("body_").unwrap_or_default().to_string()
}

fn context(state: &State) -> minijinja::Value {
    println!("context() called");
//    serde_json::json!(["3", "2", "1"])
//    vec!["3", "2", "1"]
    state.lookup("context_").unwrap_or_default()
}

fn new_jinja_env() -> Environment<'static> {
    println!("new_jinja_env");
    let mut env = Environment::new();

    env.add_function("env", get_env);
    env.add_function("substring", substring);

    // !! Standard string manipulation
    // env.add_function("trim", trim);
    env.add_function("base64_encode", base64_encode);
    env.add_function("base64url_encode", base64url_encode);
    env.add_function("base64_decode", base64_decode);
    env.add_function("base64url_decode", base64url_decode);
    env.add_function("replace_with_random", replace_with_random);
    env.add_function("raw_string", raw_string);
    //        env.add_function("word_count", word_count);

    // !! Envoy context accessors
    env.add_function("header", header);
    env.add_function("request_header", request_header);
    // env.add_function("extraction", extraction);
    env.add_function(BODY, body);
    // env.add_function("dynamic_metadata", dynamic_metadata);

    // !! Datasource Puller needed
    // env.add_function("data_source", data_source);

    // !! Requires being in an upstream filter
    // env.add_function("host_metadata", host_metadata);
    // env.add_function("cluster_metadata", cluster_metadata);

    // !! Possibly not relevant old inja internal debug stuff
    env.add_function(CONTEXT, context);

    // specific.extend(self.route_specific.into_iter());

    env
}

fn render(
    env: &Environment<'static>,
    ctx: &minijinja::Value,
    template: &str,
    parsed_body_as_json: bool,
) -> Result<String> {
    let tmpl = env
        .template_from_str(template)
        .with_context(|| format!("error creating jinja template {}", template))?;
    //    if !parsed_body_as_json && !tmpl.undeclared_variables(false).is_empty() {
    if !parsed_body_as_json {
        let undeclared_variables = tmpl.undeclared_variables(true);
        if !undeclared_variables.is_empty() {
            for v in &undeclared_variables {
                println!("calling GLOBALS_LOOKUP");
                if !GLOBALS_LOOKUP.contains(v.as_str()) {
                    return Err(TransformationError::UndeclaredJsonVariables(format!(
                        "{:?} from template {}",
                        undeclared_variables, template
                    ))
                    .into());
                }
            }
        }
    }
    tmpl.render(ctx)
        .with_context(|| format!("error rendering jinja template {}", template))
}

fn combine_errors(msg: &str, errors: Vec<Error>) -> Result<()> {
    if !errors.is_empty() {
        let combined = errors
            .into_iter()
            .map(|e| { e.chain().map(|cause| cause.to_string()).collect::<Vec<String>>().join(":")})
            .collect::<Vec<_>>()
            .join("; ");
        return Err(anyhow::anyhow!("{}: {}", msg, combined));
    }

    Ok(())
}

/// Transform Request
///
/// On any header rendering errors, we will remove the header and continue
/// All the errors are collected and bubble up the chain so they can be logged
/// On body parsing as json error, we return error immediately so we can send a
/// 400 response back
pub fn transform_request<T: TransformationOps>(
    transform: &LocalTransform,
    request_headers_map: &HashMap<String, String>,
    mut ops: T,
) -> Result<()> {
    println!("transform_request");
    let env = &*ENV;
    let mut errors = Vec::new();

    //    let mut m = BTreeMap::new();
    let mut m = HashMap::new();
    // for request rendering, both the header() and request_header() use the request_headers
    // so, setting both to the request_headers_map in the context
    m.insert(
        "headers".to_string(),
        minijinja::Value::from_serialize(request_headers_map),
    );
    m.insert(
        "request_headers".to_string(),
        minijinja::Value::from_serialize(request_headers_map),
    );
    let mut parsed_body_as_json = false;
    if let Some(body_transform) = transform.body.as_ref() {
        if matches!(body_transform.parse_as, BodyParseBehavior::AsJson) {
            println!("body_transform: {}", body_transform.value);
            let json_body = ops.parse_request_json_body()?;

            if json_body != JsonValue::Null {
                println!("body_transform: got json body");
                println!("request check add context() body_transform: {}", body_transform.value);
                if body_transform.value.contains("context()") {
                    println!("adding context_");
                    m.insert(
                        CONTEXT.to_string(),
                        minijinja::Value::from_object(&json_body),
                    );
                }

                if let JsonValue::Object(map) = json_body {
                    for (k, v) in map {
                        println!(
                            "body_transform: {} = {}",
                            k,
                            minijinja::Value::from_serialize(&v)
                        );
                        m.insert(k, minijinja::Value::from_serialize(&v));
                    }
                }

                parsed_body_as_json = true;
            }
        }
    }

    if let Some(body_transform) = transform.body.as_ref() {
        println!("request check add body() body_transform: {}", body_transform.value);
        if body_transform.value.contains("body()") {
            let body = ops.get_request_body();
            println!("adding body_");
            m.insert("body_".to_string(), minijinja::Value::from_serialize(String::from_utf8_lossy(&body)));
        }
    }

    let ctx = minijinja::Value::from(m);

    if let Some(body_transform) = transform.body.as_ref() {
        if !body_transform.value.is_empty() {
            ops.drain_request_body(u64::MAX.try_into().unwrap());
            let rendered = match render(env, &ctx, &body_transform.value, parsed_body_as_json) {
                Ok(str) => Some(str),
                Err(e) => {
                    errors.push(e);
                    None
                }
            };
            if rendered.as_deref().is_some_and(|s| !s.is_empty()) {
                let rendered_body = rendered.as_deref().unwrap().as_bytes();
                ops.set_request_header(
                    "content-length",
                    rendered_body.len().to_string().as_bytes(),
                );
                ops.append_request_body(rendered_body);
            } else {
                ops.set_request_header("content-length", b"0");
                // In classic transformation, we remove content-type only when "passthrough_body"
                // is set to true (even the body is not transformed but it comes in as 0 bytes)
                // Here, we are only removing content-type if we have an override that ended up
                // removing the body as we don't have passthrough_body setting in kgateway
                ops.remove_request_header("content-type");
            }
        }
    }

    let mut abort_processing = false;
    for NameValuePair { name: key, value } in &transform.set {
        if value.is_empty() {
            // This is following the classic transformation filter behavior
            ops.remove_request_header(key);
            continue;
        }
        let rendered = match render(env, &ctx, value, parsed_body_as_json) {
            Ok(str) => Some(str),
            Err(err) => {
                if let Some(e) = err.downcast_ref::<TransformationError>() {
                    match e {
                        TransformationError::UndeclaredJsonVariables(_) => {
                            abort_processing = true;
                        }
                    }
                }
                errors.push(err);
                None
            }
        };

        if abort_processing {
            return Err(errors.pop().unwrap());
        }

        if rendered.as_deref().is_some_and(|s| !s.is_empty()) {
            ops.set_request_header(key, rendered.as_deref().unwrap().as_bytes());
        } else {
            ops.remove_request_header(key);
        }
    }

    // TODO: "add" header is not supported by the rust SDK yet

    for key in &transform.remove {
        ops.remove_request_header(key);
    }

    combine_errors("transform_request()", errors)
}

/// Transform Response
///
/// On any rendering errors, we will remove the header and continue
/// All the errors are collected and bubble up the chain so they can be logged
pub fn transform_response<T: TransformationOps>(
    transform: &LocalTransform,
    request_headers_map: &HashMap<String, String>,
    response_headers_map: &HashMap<String, String>,
    mut ops: T,
) -> Result<()> {
    println!("transform_response");
    let env = &*ENV;
    let mut errors = Vec::new();

    let mut m = BTreeMap::new();
    // for response rendering, header() uses response_headers and request_header()
    // uses the request_headers. So, setting them in the context accordingly
    m.insert(
        "headers".to_string(),
        minijinja::Value::from_serialize(response_headers_map),
    );
    m.insert(
        "request_headers".to_string(),
        minijinja::Value::from_serialize(request_headers_map),
    );
    let mut parsed_body_as_json = false;
    if let Some(body_transform) = transform.body.as_ref() {
        if matches!(body_transform.parse_as, BodyParseBehavior::AsJson) {
            println!("body_transform: {}", body_transform.value);
            let json_body = ops.parse_response_json_body()?;

            if json_body != JsonValue::Null {
                println!("body_transform: got json body");
                println!("response check add context() body_transform: {}", body_transform.value);
                if body_transform.value.contains("context()") {
                    println!("adding context_");
                    m.insert(
                        "context_".to_string(),
                        minijinja::Value::from_serialize(&json_body),
                    );
                }

                if let JsonValue::Object(map) = json_body {
                    for (k, v) in map {
                        println!(
                            "body_transform: {} = {}",
                            k,
                            minijinja::Value::from_serialize(&v)
                        );
                        m.insert(k, minijinja::Value::from_serialize(&v));
                    }
                }
                parsed_body_as_json = true;
            }
        }
    }

    if let Some(body_transform) = transform.body.as_ref() {
        println!("response check add body() body_transform: {}", body_transform.value);
        if body_transform.value.contains("body()") {
            println!("adding body_");
            let body = ops.get_response_body();
            m.insert("body_".to_string(), minijinja::Value::from_serialize(String::from_utf8_lossy(&body)));
        }
    }

    let ctx = minijinja::Value::from(m);

    if let Some(body_transform) = transform.body.as_ref() {
        if !body_transform.value.is_empty() {
            // The envoy sdk function would drain all the bytes if the number passed in is greater
            // than the content length. This is to avoid having to iterate through the buffer to
            // calculate the size.
            ops.drain_response_body(u64::MAX.try_into().unwrap());
            let rendered = match render(env, &ctx, &body_transform.value, parsed_body_as_json) {
                Ok(str) => Some(str),
                Err(e) => {
                    errors.push(e);
                    None
                }
            };
            if rendered.as_deref().is_some_and(|s| !s.is_empty()) {
                let rendered_body = rendered.as_deref().unwrap().as_bytes();
                ops.set_response_header(
                    "content-length",
                    rendered_body.len().to_string().as_bytes(),
                );
                ops.append_response_body(rendered_body);
            } else {
                ops.set_response_header("content-length", b"0");
                // In classic transformation, we remove content-type only when "passthrough_body"
                // is set to true (even the body is not transformed but it comes in as 0 bytes)
                // Here, we are only removing content-type if we have an override that ended up
                // removing the body as we don't have passthrough_body setting in kgateway
                ops.remove_response_header("content-type");
            }
        }
    }

    let mut abort_processing = false;
    for NameValuePair { name: key, value } in &transform.set {
        if value.is_empty() {
            // This is following the classic transformation filter behavior
            ops.remove_response_header(key);
            continue;
        }
        let rendered = match render(env, &ctx, value, parsed_body_as_json) {
            Ok(str) => Some(str),
            Err(err) => {
                if let Some(e) = err.downcast_ref::<TransformationError>() {
                    match e {
                        TransformationError::UndeclaredJsonVariables(_) => {
                            abort_processing = true;
                        }
                    }
                }
                errors.push(err);
                None
            }
        };

        if abort_processing {
            return Err(errors.pop().unwrap());
        }

        if rendered.as_deref().is_some_and(|s| !s.is_empty()) {
            ops.set_response_header(key, rendered.as_deref().unwrap().as_bytes());
        } else {
            ops.remove_response_header(key);
        }
    }

    // TODO: "add" header is not supported by the rust SDK yet

    for key in &transform.remove {
        ops.remove_response_header(key);
    }

    combine_errors("transform_response()", errors)
}
