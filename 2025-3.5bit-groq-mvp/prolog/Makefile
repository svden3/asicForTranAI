# Makefile for Prolog Inference Engine
# Target: Groq LPU / Cerebras WSE via MLIR

FC = gfortran
FFLAGS = -std=f2023 -Wall -Wextra -O2 -g -fcheck=all
LDFLAGS = -O2

SRC_DIR = .
BIN_DIR = bin

SOURCES = prolog_engine.f90 test_prolog_engine.f90
OBJECTS = $(BIN_DIR)/prolog_engine.o $(BIN_DIR)/test_prolog_engine.o
TARGET = $(BIN_DIR)/test_prolog

.PHONY: all clean test help

all: $(TARGET)

help:
	@echo "Prolog Inference Engine Build System"
	@echo "======================================"
	@echo ""
	@echo "Targets:"
	@echo "  make all      - Build the Prolog engine"
	@echo "  make test     - Build and run tests"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make help     - Show this message"
	@echo ""
	@echo "Requirements:"
	@echo "  - gfortran with Fortran 2023 support"
	@echo ""

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/prolog_engine.o: prolog_engine.f90 | $(BIN_DIR)
	$(FC) $(FFLAGS) -c $< -o $@

$(BIN_DIR)/test_prolog_engine.o: test_prolog_engine.f90 $(BIN_DIR)/prolog_engine.o | $(BIN_DIR)
	$(FC) $(FFLAGS) -c $< -o $@ -I$(BIN_DIR)

$(TARGET): $(OBJECTS)
	$(FC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo ""
	@echo "Build complete: $(TARGET)"
	@echo ""

test: $(TARGET)
	@echo ""
	@echo "Running Prolog engine tests..."
	@echo ""
	$(TARGET)
	@echo ""

clean:
	rm -rf $(BIN_DIR)
	rm -f *.mod *.o *~
	@echo "Clean complete"
