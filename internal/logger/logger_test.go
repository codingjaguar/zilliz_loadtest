package logger

import (
	"os"
	"testing"
)

func TestParseLevel(t *testing.T) {
	tests := []struct {
		name  string
		level string
		want  string
	}{
		{"DEBUG", "DEBUG", "DEBUG"},
		{"INFO", "INFO", "INFO"},
		{"WARN", "WARN", "WARN"},
		{"ERROR", "ERROR", "ERROR"},
		{"lowercase", "info", "INFO"},
		{"default", "INVALID", "INFO"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			Init(tt.level, "text")
			// Just verify it doesn't panic
		})
	}
}

func TestPrintf(t *testing.T) {
	Init("INFO", "text")
	Printf("test message %d", 123)
	// Just verify it doesn't panic
}

func TestPrint(t *testing.T) {
	Init("INFO", "text")
	Print("test message")
	// Just verify it doesn't panic
}

func TestPrintln(t *testing.T) {
	Init("INFO", "text")
	Println("test message")
	// Just verify it doesn't panic
}

func TestFprintf(t *testing.T) {
	Init("INFO", "text")
	Fprintf(os.Stderr, "error message %d", 456)
	// Just verify it doesn't panic
}
