# SPDX-FileCopyrightText: 2026 ixsimpl contributors
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.17)
project(ixsimpl C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(USE_AMALGAMATION "Build from single-file ixsimpl_amalg.c" OFF)

if(USE_AMALGAMATION)
    set(LIB_SOURCES ixsimpl_amalg.c)
else()
    set(LIB_SOURCES
        src/arena.c
        src/bounds.c
        src/ctx.c
        src/expand.c
        src/interval.c
        src/node.c
        src/parser.c
        src/print.c
        src/rational.c
        src/simplify.c
        src/walk.c
    )
endif()

# --- Python extension (driven by scikit-build-core) ---

if(SKBUILD)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    python_add_library(_ixsimpl MODULE WITH_SOABI
        bindings/python/_ixsimpl.c
        ${LIB_SOURCES}
    )
    target_include_directories(_ixsimpl PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    set_target_properties(_ixsimpl PROPERTIES C_VISIBILITY_PRESET hidden)

    install(TARGETS _ixsimpl DESTINATION ixsimpl)
    install(FILES bindings/python/ixsimpl/_ixsimpl.pyi DESTINATION ixsimpl)
    return()
endif()

# --- Standalone C library + tests + benchmarks ---

add_compile_options(-pedantic -Wall -Wextra -Werror)

option(ENABLE_ASAN "Build with AddressSanitizer" OFF)
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

option(ENABLE_STATS "Build with rule-hit statistics (IXS_STATS)" OFF)
if(ENABLE_STATS)
    add_compile_definitions(IXS_STATS)
endif()

add_library(ixsimpl STATIC ${LIB_SOURCES})
target_include_directories(ixsimpl
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# --- Tests ---

enable_testing()

# Tests that use only the public API — always built.

add_executable(test_parser test/test_parser.c)
target_link_libraries(test_parser ixsimpl)
add_test(NAME test_parser COMMAND test_parser)

# Stats-enabled build: test_simplify links this to collect rule hit counts.
add_library(ixsimpl_stats STATIC ${LIB_SOURCES})
target_include_directories(ixsimpl_stats
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(ixsimpl_stats PRIVATE IXS_STATS)

add_executable(test_simplify test/test_simplify.c)
target_link_libraries(test_simplify ixsimpl_stats)
add_test(NAME test_simplify COMMAND test_simplify)

add_executable(test_corpus test/test_corpus.c)
target_link_libraries(test_corpus ixsimpl)
add_test(NAME test_corpus COMMAND test_corpus
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test_corpus_gemm256x192 COMMAND test_corpus
         --corpus=test/corpus_gemm256x192.txt
         --assumptions=test/corpus_gemm256x192_assumptions.txt
         --expected=test/corpus_gemm256x192_expected.txt
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(test_edge_cases test/test_edge_cases.c)
target_link_libraries(test_edge_cases ixsimpl)
add_test(NAME test_edge_cases COMMAND test_edge_cases)

add_executable(test_introspect test/test_introspect.c)
target_link_libraries(test_introspect ixsimpl)
add_test(NAME test_introspect COMMAND test_introspect)

add_executable(test_expand test/test_expand.c)
target_link_libraries(test_expand ixsimpl)
add_test(NAME test_expand COMMAND test_expand)

# Tests that need internal symbols — skipped in amalgamated builds.

if(NOT USE_AMALGAMATION)
    add_executable(test_rational test/test_rational.c)
    target_link_libraries(test_rational ixsimpl)
    target_include_directories(test_rational PRIVATE src)
    add_test(NAME test_rational COMMAND test_rational)

    add_executable(test_arena test/test_arena.c)
    target_link_libraries(test_arena ixsimpl)
    target_include_directories(test_arena PRIVATE src)
    add_test(NAME test_arena COMMAND test_arena)

    add_executable(test_bounds test/test_bounds.c)
    target_link_libraries(test_bounds ixsimpl)
    target_include_directories(test_bounds PRIVATE src)
    add_test(NAME test_bounds COMMAND test_bounds)
endif()

# --- Benchmark ---

add_executable(bench_corpus bench/bench_corpus.c)
target_link_libraries(bench_corpus ixsimpl)
