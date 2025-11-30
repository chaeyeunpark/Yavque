function(get_blas_version OUT_VAR)
    if(NOT BLAS_LIBRARIES)
        message(FATAL_ERROR "BLAS_LIBRARIES variable is not set.")
    endif()

    # Use the first library from BLAS_LIBRARIES
    list(GET BLAS_LIBRARIES 0 BLAS_LIB)

    if(NOT EXISTS "${BLAS_LIB}")
        message(FATAL_ERROR "BLAS library does not exist: ${BLAS_LIB}")
    endif()

    set(VERSION "")

    #
    # ---- Try OpenBLAS ----
    #
    if(NOT VERSION)
        if(BLAS_LIB MATCHES ".*libopenblas.*")
            set(VERSION "OpenBLAS")
        endif()
    endif()

    #
    # ---- Try Intel MKL ----
    #
    if(NOT VERSION)
        if(BLAS_LIB MATCHES ".*libmkl.*")
            set(VERSION "MKL")
        endif()
    endif()

    if(NOT VERSION)
        set(VERSION "unknown")
    endif()

    set(${OUT_VAR} "${VERSION}" PARENT_SCOPE)
endfunction()
