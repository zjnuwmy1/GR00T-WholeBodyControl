# Like GNU `make`, but `just` rustier.
# https://just.systems/
# run `just` from this directory to see available commands

alias b := build
alias r := run
alias t := test
alias c := clean
alias ch := check

# Default command when 'just' is run without arguments
default:
  @just --list

# Get the number of cores
CORES := if os() == "macos" { `sysctl -n hw.ncpu` } else if os() == "linux" { `nproc` } else { "1" }

# Build the project
# CMAKE_EXTRA_ARGS is passed through to configure, so a caller can select the
# inference backend without editing this file, e.g.
#   CMAKE_EXTRA_ARGS=-DUSE_TENSORRT=OFF just build
# Passing it explicitly (rather than relying on the CMake cache) keeps the choice
# correct after `just clean` wipes build/.
build *build_type='Release':
  @mkdir -p build
  @echo "Configuring the build system..."
  @cd build && cmake -S .. -B . -DCMAKE_BUILD_TYPE={{build_type}} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ${CMAKE_EXTRA_ARGS:-}
  @echo "Building the project..."
  @cd build && cmake --build . -j{{CORES}}

# Run a package
run *package='hello':
  @./target/release/{{package}}

# Run code quality tools
test:
  @echo "Running tests..."

# Remove build artifacts and non-essential files
clean:
  @echo "Cleaning..."
  @rm -rf build
  @rm -rf target

# Run code quality tools
check:
  @echo "Running code quality tools..."
  @cppcheck --error-exitcode=1 --project=build/compile_commands.json -i build/_deps/

