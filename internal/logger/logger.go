package logger

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"
)

var (
	defaultLogger *slog.Logger
	logLevel      slog.Level = slog.LevelInfo
	logFormat     string     = "text"
)

// Init initializes the logger with the specified level and format
func Init(level string, format string) {
	logLevel = parseLevel(level)
	logFormat = format

	var handler slog.Handler
	writer := os.Stdout

	if format == "json" {
		opts := &slog.HandlerOptions{
			Level: logLevel,
			ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
				// Customize JSON output
				if a.Key == slog.TimeKey {
					return slog.String("timestamp", a.Value.Time().Format(time.RFC3339))
				}
				return a
			},
		}
		handler = slog.NewJSONHandler(writer, opts)
	} else {
		opts := &slog.HandlerOptions{
			Level: logLevel,
		}
		handler = slog.NewTextHandler(writer, opts)
	}

	defaultLogger = slog.New(handler)
}

// parseLevel parses a log level string
func parseLevel(level string) slog.Level {
	switch strings.ToUpper(level) {
	case "DEBUG":
		return slog.LevelDebug
	case "INFO":
		return slog.LevelInfo
	case "WARN", "WARNING":
		return slog.LevelWarn
	case "ERROR":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// Logger wraps slog.Logger with convenience methods
type Logger struct {
	*slog.Logger
}

// GetLogger returns the default logger
func GetLogger() *Logger {
	if defaultLogger == nil {
		Init("INFO", "text")
	}
	return &Logger{defaultLogger}
}

// WithRequestID adds a request ID to the logger context
func (l *Logger) WithRequestID(requestID string) *Logger {
	return &Logger{l.Logger.With("request_id", requestID)}
}

// Debug logs a debug message
func Debug(msg string, args ...any) {
	GetLogger().Debug(msg, args...)
}

// Info logs an info message
func Info(msg string, args ...any) {
	GetLogger().Info(msg, args...)
}

// Warn logs a warning message
func Warn(msg string, args ...any) {
	GetLogger().Warn(msg, args...)
}

// Error logs an error message
func Error(msg string, args ...any) {
	GetLogger().Error(msg, args...)
}

// DebugContext logs a debug message with context
func DebugContext(ctx context.Context, msg string, args ...any) {
	GetLogger().DebugContext(ctx, msg, args...)
}

// InfoContext logs an info message with context
func InfoContext(ctx context.Context, msg string, args ...any) {
	GetLogger().InfoContext(ctx, msg, args...)
}

// WarnContext logs a warning message with context
func WarnContext(ctx context.Context, msg string, args ...any) {
	GetLogger().WarnContext(ctx, msg, args...)
}

// ErrorContext logs an error message with context
func ErrorContext(ctx context.Context, msg string, args ...any) {
	GetLogger().ErrorContext(ctx, msg, args...)
}

// Printf provides compatibility with fmt.Printf for gradual migration
func Printf(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	// Remove trailing newline if present (slog adds its own)
	msg = strings.TrimSuffix(msg, "\n")
	GetLogger().Info(msg)
}

// Print provides compatibility with fmt.Print
func Print(args ...any) {
	msg := fmt.Sprint(args...)
	msg = strings.TrimSuffix(msg, "\n")
	GetLogger().Info(msg)
}

// Println provides compatibility with fmt.Println
func Println(args ...any) {
	msg := fmt.Sprintln(args...)
	msg = strings.TrimSuffix(msg, "\n")
	GetLogger().Info(msg)
}

// Fprintf provides compatibility with fmt.Fprintf (for stderr)
func Fprintf(w io.Writer, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	msg = strings.TrimSuffix(msg, "\n")
	if w == os.Stderr {
		GetLogger().Error(msg)
	} else {
		GetLogger().Info(msg)
	}
}

// SetOutput is a no-op for compatibility (slog uses handlers)
func SetOutput(w io.Writer) {
	// No-op - slog uses handlers
}

// ProgressLogger provides structured progress logging
type ProgressLogger struct {
	logger *Logger
	prefix string
}

// NewProgressLogger creates a new progress logger
func NewProgressLogger(prefix string) *ProgressLogger {
	return &ProgressLogger{
		logger: GetLogger(),
		prefix: prefix,
	}
}

// Log logs a progress message
func (pl *ProgressLogger) Log(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	pl.logger.Info(msg, "prefix", pl.prefix, "type", "progress")
}

// Logf logs a formatted progress message
func (pl *ProgressLogger) Logf(format string, args ...any) {
	pl.Log(format, args...)
}
