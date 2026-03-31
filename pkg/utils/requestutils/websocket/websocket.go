package websocket

import (
	"net/http"
	"time"

	gorillaws "github.com/gorilla/websocket"
)

// Dial establishes a WebSocket connection to the given URL, using the provided
// Host header and optional extra headers. It sets the given deadline on the
// dialer so that hanging connections (e.g. due to Envoy body buffering blocking
// the upgrade) are detected quickly.
//
// On success it returns the text of the first message sent by the server.
func Dial(url, host string, deadline time.Duration, extraHeaders http.Header) (string, error) {
	dialer := gorillaws.Dialer{
		HandshakeTimeout: deadline,
	}

	reqHeader := http.Header{
		"Host": []string{host},
	}
	for k, v := range extraHeaders {
		reqHeader[k] = v
	}

	conn, _, err := dialer.Dial(url, reqHeader)
	if err != nil {
		return "", err
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(deadline)) //nolint:errcheck
	_, msg, err := conn.ReadMessage()
	if err != nil {
		return "", err
	}
	return string(msg), nil
}
