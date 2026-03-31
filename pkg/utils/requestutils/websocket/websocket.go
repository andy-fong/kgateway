package websocket

import (
	"fmt"
	"net/http"
	"time"

	gorillaws "github.com/gorilla/websocket"
)

// Dial establishes a WebSocket connection to the given URL, using the provided
// Host header and optional extra headers. It sets the given deadline on the
// dialer so that hanging connections (e.g. due to Envoy body buffering blocking
// the upgrade) are detected quickly.
//
// After a successful handshake it sends a short test message and returns the
// echoed response. This works with echo-style WebSocket servers (like
// jmalloc/echo-server) that only reply after receiving a client message.
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
		return "", fmt.Errorf("websocket handshake failed: %w", err)
	}
	defer conn.Close()

	// Echo servers don't send a greeting — send a message first.
	const testPayload = "websocket-e2e-ping"
	conn.SetWriteDeadline(time.Now().Add(deadline)) //nolint:errcheck
	if err := conn.WriteMessage(gorillaws.TextMessage, []byte(testPayload)); err != nil {
		return "", fmt.Errorf("websocket write failed: %w", err)
	}

	conn.SetReadDeadline(time.Now().Add(deadline)) //nolint:errcheck
	_, msg, err := conn.ReadMessage()
	if err != nil {
		return "", fmt.Errorf("websocket read failed: %w", err)
	}
	return string(msg), nil
}
