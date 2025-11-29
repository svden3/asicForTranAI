# Makefile for 3.5-bit Groq ASIC Inference
# Target: 4188 tok/s on Groq LPU for 70B model

# Compiler settings
FC = gfortran
FFLAGS = -O3 -march=native -ffast-math -funroll-loops
GROQ_FLAGS = # TODO: Add Groq-specific compilation flags

# Source files
SOURCES = matmul_template.f90
OBJECTS = $(SOURCES:.f90=.o)
TARGET = groq_inference

# Default target
all: $(TARGET)
	@echo "Built 3.5-bit Groq inference binary"
	@echo "TODO: Add actual Groq ASIC deployment steps"

# Build executable
$(TARGET): $(OBJECTS)
	$(FC) $(FFLAGS) -o $@ $^

# Compile Fortran sources
%.o: %.f90
	$(FC) $(FFLAGS) -c $<

# Run benchmarks
benchmark: $(TARGET)
	@echo "Running performance benchmark..."
	@echo "Target: 4188 tok/s on Groq LPU"
	./$(TARGET)
	# TODO: Add actual Groq benchmark commands

# Deploy to Groq ASIC
deploy:
	@echo "Deploying to Groq LPU..."
	# TODO: Add Groq deployment commands
	# TODO: Configure ASIC parameters
	@echo "Replace with actual Groq deployment script"

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) *.mod

# Run tests
test: $(TARGET)
	@echo "Running tests..."
	# TODO: Add test suite
	./$(TARGET)

.PHONY: all benchmark deploy clean test
