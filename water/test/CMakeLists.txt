configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(WATER_TEST_DEPENDS
        FileCheck count not
        water-opt
        )
add_lit_testsuite(check-water "Running the water regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${WATER_TEST_DEPENDS}
        )
set_target_properties(check-water PROPERTIES FOLDER "Tests")

add_lit_testsuites(WATER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${WATER_TEST_DEPENDS})

add_subdirectory(lib)
