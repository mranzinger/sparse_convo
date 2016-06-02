package = "sparseconvo"
version = "scm-1"

source = {
    url = "https://github.com/mranzinger/sparse_convo.git"
}

description = {
    summary = "Torch Sparse Filter Convolution Implementation",
    detailed = [[
    ]],
    homepage = "https://github.com/mranzinger/sparse_convo",
    license = "BSD"
}

dependencies = {
    "torch >= 7.0",
    "nn >= 1.0",
    "cutorch >= 1.0",
    "multikey"
}

build = {
    type = "command",
    build_command = [[
        cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) -j$(getconf _NPROCESSORS_ONLN) install
        ]],
        install_command = "cd build"
}
