use serde::Deserialize;
use serde::de::{self, Deserializer};
use serde_json::Value;
type Strng = String;

pub mod jinja;

#[derive(Default, Clone, Deserialize)]
pub struct LocalTransformationConfig {
    #[serde(default)]
    pub request: Option<LocalTransform>,
    #[serde(default)]
    pub response: Option<LocalTransform>,
}

#[derive(Default, Clone, Deserialize)]
pub struct LocalTransform {
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_name_value")]
    pub add: Vec<(Strng, Strng)>,
    #[serde(default)]
    #[serde(deserialize_with = "deserialize_name_value")]
    pub set: Vec<(Strng, Strng)>,
    #[serde(default)]
    pub remove: Vec<Strng>,
    #[serde(default)]
    pub body: Option<BodyTransform>,
}

#[derive(Default, Clone, Deserialize)]
pub struct BodyTransform {
    #[serde(default)]
    pub parseAs: Strng,
    #[serde(default)]
    pub value: String
}


fn deserialize_name_value<'de, D>(deserializer: D) -> Result<Vec<(Strng, Strng)>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw: Vec<Value> = Deserialize::deserialize(deserializer)?;
    let mut result = Vec::new();

    for item in raw {
        if let (Some(name), Some(value)) = (item.get("name"), item.get("value")) {
            result.push(
                (name.as_str().unwrap().to_string(),
                value.as_str().unwrap().to_string())
            );
        } else {
            return Err(de::Error::custom("missing name or value in header item"));
        }
    }

    Ok(result)
}