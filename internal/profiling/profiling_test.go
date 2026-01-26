package profiling

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestStartCPUProfiling(t *testing.T) {
	tmpDir := t.TempDir()
	profilePath := filepath.Join(tmpDir, "cpu_test.pprof")

	file, err := StartCPUProfiling(profilePath)
	if err != nil {
		t.Fatalf("StartCPUProfiling() error = %v", err)
	}
	defer file.Close()

	// Verify file exists
	if _, err := os.Stat(profilePath); os.IsNotExist(err) {
		t.Errorf("StartCPUProfiling() file does not exist: %s", profilePath)
	}

	// Stop profiling
	err = StopCPUProfiling(file)
	if err != nil {
		t.Errorf("StopCPUProfiling() error = %v", err)
	}
}

func TestStartCPUProfiling_DefaultPath(t *testing.T) {
	tmpDir := t.TempDir()
	// Change to temp dir to test default path generation
	oldDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldDir)

	file, err := StartCPUProfiling("")
	if err != nil {
		t.Fatalf("StartCPUProfiling() error = %v", err)
	}
	defer file.Close()

	// Check that a file was created with timestamp pattern
	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read directory: %v", err)
	}

	found := false
	for _, f := range files {
		if contains(f.Name(), "cpu_profile_") && contains(f.Name(), ".pprof") {
			found = true
			// Clean up
			os.Remove(f.Name())
			break
		}
	}
	if !found {
		t.Error("StartCPUProfiling() should create file with timestamp pattern")
	}

	err = StopCPUProfiling(file)
	if err != nil {
		t.Errorf("StopCPUProfiling() error = %v", err)
	}
}

func TestStopCPUProfiling(t *testing.T) {
	tmpDir := t.TempDir()
	profilePath := filepath.Join(tmpDir, "cpu_test.pprof")

	file, err := StartCPUProfiling(profilePath)
	if err != nil {
		t.Fatalf("StartCPUProfiling() error = %v", err)
	}

	err = StopCPUProfiling(file)
	if err != nil {
		t.Errorf("StopCPUProfiling() error = %v", err)
	}

	// Verify file is closed
	if _, err := file.Write([]byte("test")); err == nil {
		t.Error("StopCPUProfiling() file should be closed")
	}
}

func TestStopCPUProfiling_NilFile(t *testing.T) {
	err := StopCPUProfiling(nil)
	if err != nil {
		t.Errorf("StopCPUProfiling(nil) error = %v, want nil", err)
	}
}

func TestWriteMemoryProfile(t *testing.T) {
	tmpDir := t.TempDir()
	profilePath := filepath.Join(tmpDir, "mem_test.pprof")

	err := WriteMemoryProfile(profilePath)
	if err != nil {
		t.Fatalf("WriteMemoryProfile() error = %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(profilePath); os.IsNotExist(err) {
		t.Errorf("WriteMemoryProfile() file does not exist: %s", profilePath)
	}

	// Verify file is not empty
	info, err := os.Stat(profilePath)
	if err != nil {
		t.Fatalf("Failed to stat file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("WriteMemoryProfile() file should not be empty")
	}
}

func TestWriteMemoryProfile_DefaultPath(t *testing.T) {
	tmpDir := t.TempDir()
	oldDir, _ := os.Getwd()
	os.Chdir(tmpDir)
	defer os.Chdir(oldDir)

	err := WriteMemoryProfile("")
	if err != nil {
		t.Fatalf("WriteMemoryProfile() error = %v", err)
	}

	// Check that a file was created with timestamp pattern
	files, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatalf("Failed to read directory: %v", err)
	}

	found := false
	for _, f := range files {
		if contains(f.Name(), "mem_profile_") && contains(f.Name(), ".pprof") {
			found = true
			os.Remove(f.Name())
			break
		}
	}
	if !found {
		t.Error("WriteMemoryProfile() should create file with timestamp pattern")
	}
}

func TestGetMemStats(t *testing.T) {
	stats := GetMemStats()

	if stats.Alloc == 0 && stats.TotalAlloc == 0 {
		t.Error("GetMemStats() should return non-zero values")
	}

	// Verify it's a valid MemStats
	if stats.NumGC < 0 {
		t.Error("GetMemStats() NumGC should be non-negative")
	}
}

func TestFormatMemStats(t *testing.T) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	formatted := FormatMemStats(m)

	if formatted == "" {
		t.Error("FormatMemStats() should return non-empty string")
	}

	// Check that it contains expected fields
	if !contains(formatted, "Alloc") {
		t.Error("FormatMemStats() should contain 'Alloc'")
	}
	if !contains(formatted, "TotalAlloc") {
		t.Error("FormatMemStats() should contain 'TotalAlloc'")
	}
	if !contains(formatted, "Sys") {
		t.Error("FormatMemStats() should contain 'Sys'")
	}
	if !contains(formatted, "NumGC") {
		t.Error("FormatMemStats() should contain 'NumGC'")
	}
}

func TestProfilingConfig(t *testing.T) {
	config := ProfilingConfig{
		CPUProfile:    "cpu.pprof",
		MemoryProfile: "mem.pprof",
		Enabled:       true,
	}

	if config.CPUProfile != "cpu.pprof" {
		t.Errorf("ProfilingConfig.CPUProfile = %v, want cpu.pprof", config.CPUProfile)
	}
	if config.MemoryProfile != "mem.pprof" {
		t.Errorf("ProfilingConfig.MemoryProfile = %v, want mem.pprof", config.MemoryProfile)
	}
	if !config.Enabled {
		t.Error("ProfilingConfig.Enabled = false, want true")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || 
			(len(s) > len(substr) && 
				(s[:len(substr)] == substr || 
					s[len(s)-len(substr):] == substr ||
					containsSubstring(s, substr))))
}

func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
