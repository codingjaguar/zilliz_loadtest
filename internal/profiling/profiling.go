package profiling

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

// ProfilingConfig holds profiling configuration
type ProfilingConfig struct {
	CPUProfile    string
	MemoryProfile string
	Enabled       bool
}

// StartCPUProfiling starts CPU profiling
func StartCPUProfiling(profilePath string) (*os.File, error) {
	if profilePath == "" {
		profilePath = fmt.Sprintf("cpu_profile_%s.pprof", time.Now().Format("20060102_150405"))
	}

	f, err := os.Create(profilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to create CPU profile file: %w", err)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("failed to start CPU profiling: %w", err)
	}

	return f, nil
}

// StopCPUProfiling stops CPU profiling
func StopCPUProfiling(f *os.File) error {
	pprof.StopCPUProfile()
	if f != nil {
		return f.Close()
	}
	return nil
}

// WriteMemoryProfile writes a memory profile
func WriteMemoryProfile(profilePath string) error {
	if profilePath == "" {
		profilePath = fmt.Sprintf("mem_profile_%s.pprof", time.Now().Format("20060102_150405"))
	}

	f, err := os.Create(profilePath)
	if err != nil {
		return fmt.Errorf("failed to create memory profile file: %w", err)
	}
	defer f.Close()

	runtime.GC() // Force garbage collection before profiling
	if err := pprof.WriteHeapProfile(f); err != nil {
		return fmt.Errorf("failed to write memory profile: %w", err)
	}

	return nil
}

// GetMemStats returns current memory statistics
func GetMemStats() runtime.MemStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m
}

// FormatMemStats formats memory statistics as a string
func FormatMemStats(m runtime.MemStats) string {
	return fmt.Sprintf("Alloc: %d KB, TotalAlloc: %d KB, Sys: %d KB, NumGC: %d",
		m.Alloc/1024, m.TotalAlloc/1024, m.Sys/1024, m.NumGC)
}
