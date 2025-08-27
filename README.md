cmake_minimum_required(VERSION 3.10)
project(mathlib VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MATH_NO_BUILD "Use prebuilt version instead of building from source" OFF)

message(STATUS "MATH_NO_BUILD = ${MATH_NO_BUILD}")

set(DEPLOY_DIR ${PROJECT_SOURCE_DIR}/surum)

if(MATH_NO_BUILD)
    message(STATUS "Using prebuilt mathlib from ${DEPLOY_DIR}")

    # Create an INTERFACE target that only points to includes + lib
    add_library(${PROJECT_NAME} INTERFACE)

    target_include_directories(${PROJECT_NAME} INTERFACE
        ${DEPLOY_DIR}/inc
    )

    target_link_libraries(${PROJECT_NAME} INTERFACE
        ${DEPLOY_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    add_library(mathlib::mathlib ALIAS mathlib)

else()
    message(STATUS "Building mathlib from sources")

    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)

    add_library(${PROJECT_NAME} STATIC src/math.cpp)

    target_include_directories(${PROJECT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
            $<INSTALL_INTERFACE:include>
    )

    add_library(mathlib::mathlib ALIAS mathlib)

    # ---- Custom post-build copy step ----
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${DEPLOY_DIR}/lib
        COMMENT "Copying ${PROJECT_NAME} library to ${DEPLOY_DIR}/lib"
    )

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${DEPLOY_DIR}/inc
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/inc ${DEPLOY_DIR}/inc
        COMMENT "Copying header files to ${DEPLOY_DIR}/inc"
    )

    # ---- Export/install setup (optional, still works) ----
    include(CMakePackageConfigHelpers)
    install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY inc/ DESTINATION include)

    install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Config.cmake
        NAMESPACE mathlib::
        DESTINATION lib/cmake/${PROJECT_NAME}
    )
endif()

