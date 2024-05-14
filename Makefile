# Default target named 'all'
all:
	@echo "Configuring and building the project..."
	cmake -B build .
	make -C build -j

# Clean target to remove build directory
clean:
	@echo "Cleaning up..."
	rm -rf build/
