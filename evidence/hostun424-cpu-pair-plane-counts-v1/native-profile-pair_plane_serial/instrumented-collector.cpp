#include <chrono>
#include <cstdint>
#include <cstring>
static thread_local uint64_t profile_calls[8] = {};
static thread_local uint64_t profile_ns[8] = {};
struct QueryScope {
    int index;
    std::chrono::steady_clock::time_point start;
    QueryScope(int i):index(i),start(std::chrono::steady_clock::now()) { ++profile_calls[i]; }
    ~QueryScope() { profile_ns[index] += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count(); }
};
#ifndef SLANG_CPP_PRELUDE_H
#define SLANG_CPP_PRELUDE_H

// Because the signature of isnan, isfinite, and is isinf changed in C++, we use the macro
// to use the version in the std namespace.
// https://stackoverflow.com/questions/39130040/cmath-hides-isnan-in-math-h-in-c14-c11

#ifdef SLANG_LLVM
#ifndef SLANG_LLVM_H
#define SLANG_LLVM_H

// TODO(JS):
// Disable exception declspecs, as not supported on LLVM without some extra options.
// We could enable with `-fms-extensions`
#define SLANG_DISABLE_EXCEPTIONS 1

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
extern "C" void assertFailure(const char* msg);
#define SLANG_PRELUDE_EXPECT(VALUE, MSG) \
    if (VALUE)                           \
    {                                    \
    }                                    \
    else                                 \
        assertFailure("assertion failed: '" MSG "'")
#define SLANG_PRELUDE_ASSERT(VALUE) SLANG_PRELUDE_EXPECT(VALUE, #VALUE)
#else // SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_EXPECT(VALUE, MSG)
#define SLANG_PRELUDE_ASSERT(x)
#endif // SLANG_PRELUDE_ENABLE_ASSERT
#endif

/*
Taken from stddef.h
*/

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __SIZE_TYPE__ rsize_t;

// typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#define NULL __null
#else
#define NULL 0
#endif
#else
#define NULL ((void*)0)
#endif
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std
{
typedef decltype(nullptr) nullptr_t;
}
using ::std::nullptr_t;
#endif
#endif
#undef __need_NULL
#endif /* defined(__need_NULL) */


/*
The following are taken verbatim from stdint.h from Clang in LLVM. Only 8/16/32/64 types are needed.
*/

// LLVM/Clang types such that we can use LLVM/Clang without headers for C++ output from Slang

#ifdef __INT64_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int64_t*/
typedef __INT64_TYPE__ int64_t;
#endif /* __int8_t_defined */
typedef __UINT64_TYPE__ uint64_t;
#define __int_least64_t int64_t
#define __uint_least64_t uint64_t
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
typedef __int_least64_t int_least64_t;
typedef __uint_least64_t uint_least64_t;
typedef __int_least64_t int_fast64_t;
typedef __uint_least64_t uint_fast64_t;
#endif /* __int_least64_t */

#ifdef __INT32_TYPE__

#ifndef __int8_t_defined /* glibc sys/types.h also defines int32_t*/
typedef __INT32_TYPE__ int32_t;
#endif /* __int8_t_defined */

#ifndef __uint32_t_defined /* more glibc compatibility */
#define __uint32_t_defined
typedef __UINT32_TYPE__ uint32_t;
#endif /* __uint32_t_defined */

#define __int_least32_t int32_t
#define __uint_least32_t uint32_t
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
typedef __int_least32_t int_least32_t;
typedef __uint_least32_t uint_least32_t;
typedef __int_least32_t int_fast32_t;
typedef __uint_least32_t uint_fast32_t;
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int16_t*/
typedef __INT16_TYPE__ int16_t;
#endif /* __int8_t_defined */
typedef __UINT16_TYPE__ uint16_t;
#define __int_least16_t int16_t
#define __uint_least16_t uint16_t
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
typedef __int_least16_t int_least16_t;
typedef __uint_least16_t uint_least16_t;
typedef __int_least16_t int_fast16_t;
typedef __uint_least16_t uint_fast16_t;
#endif /* __int_least16_t */

#ifdef __INT8_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int8_t*/
typedef __INT8_TYPE__ int8_t;
#endif /* __int8_t_defined */
typedef __UINT8_TYPE__ uint8_t;
#define __int_least8_t int8_t
#define __uint_least8_t uint8_t
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
typedef __int_least8_t int_least8_t;
typedef __uint_least8_t uint_least8_t;
typedef __int_least8_t int_fast8_t;
typedef __uint_least8_t uint_fast8_t;
#endif /* __int_least8_t */

/* prevent glibc sys/types.h from defining conflicting types */
#ifndef __int8_t_defined
#define __int8_t_defined
#endif /* __int8_t_defined */

/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#define __stdint_join3(a, b, c) a##b##c

#ifndef _INTPTR_T
#ifndef __intptr_t_defined
typedef __INTPTR_TYPE__ intptr_t;
#define __intptr_t_defined
#define _INTPTR_T
#endif
#endif

#ifndef _UINTPTR_T
typedef __UINTPTR_TYPE__ uintptr_t;
#define _UINTPTR_T
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.4 Macros for minimum-width integer constants.
 *
 * The standard requires that integer constant macros be defined for all the
 * minimum-width types defined above. As 8-, 16-, 32-, and 64-bit minimum-width
 * types are required, the corresponding integer constant macros are defined
 * here. This implementation also defines minimum-width types for every other
 * integer width that the target implements, so corresponding macros are
 * defined below, too.
 *
 * These macros are defined using the same successive-shrinking approach as
 * the type definitions above. It is likewise important that macros are defined
 * in order of decending width.
 *
 * Note that C++ should not check __STDC_CONSTANT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#define __int_c_join(a, b) a##b
#define __int_c(v, suffix) __int_c_join(v, suffix)
#define __uint_c(v, suffix) __int_c_join(v##U, suffix)

#ifdef __INT64_TYPE__
#ifdef __INT64_C_SUFFIX__
#define __int64_c_suffix __INT64_C_SUFFIX__
#else
#undef __int64_c_suffix
#endif /* __INT64_C_SUFFIX__ */
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
#ifdef __int64_c_suffix
#define INT64_C(v) __int_c(v, __int64_c_suffix)
#define UINT64_C(v) __uint_c(v, __int64_c_suffix)
#else
#define INT64_C(v) v
#define UINT64_C(v) v##U
#endif /* __int64_c_suffix */
#endif /* __int_least64_t */


#ifdef __INT32_TYPE__
#ifdef __INT32_C_SUFFIX__
#define __int32_c_suffix __INT32_C_SUFFIX__
#else
#undef __int32_c_suffix
#endif /* __INT32_C_SUFFIX__ */
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
#ifdef __int32_c_suffix
#define INT32_C(v) __int_c(v, __int32_c_suffix)
#define UINT32_C(v) __uint_c(v, __int32_c_suffix)
#else
#define INT32_C(v) v
#define UINT32_C(v) v##U
#endif /* __int32_c_suffix */
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifdef __INT16_C_SUFFIX__
#define __int16_c_suffix __INT16_C_SUFFIX__
#else
#undef __int16_c_suffix
#endif /* __INT16_C_SUFFIX__ */
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
#ifdef __int16_c_suffix
#define INT16_C(v) __int_c(v, __int16_c_suffix)
#define UINT16_C(v) __uint_c(v, __int16_c_suffix)
#else
#define INT16_C(v) v
#define UINT16_C(v) v##U
#endif /* __int16_c_suffix */
#endif /* __int_least16_t */


#ifdef __INT8_TYPE__
#ifdef __INT8_C_SUFFIX__
#define __int8_c_suffix __INT8_C_SUFFIX__
#else
#undef __int8_c_suffix
#endif /* __INT8_C_SUFFIX__ */
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
#ifdef __int8_c_suffix
#define INT8_C(v) __int_c(v, __int8_c_suffix)
#define UINT8_C(v) __uint_c(v, __int8_c_suffix)
#else
#define INT8_C(v) v
#define UINT8_C(v) v##U
#endif /* __int8_c_suffix */
#endif /* __int_least8_t */

/* C99 7.18.2.1 Limits of exact-width integer types.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 *
 * The presence of limit macros are completely optional in C99.  This
 * implementation defines limits for all of the types (exact- and
 * minimum-width) that it defines above, using the limits of the minimum-width
 * type for any types that do not have exact-width representations.
 *
 * As in the type definitions, this section takes an approach of
 * successive-shrinking to determine which limits to use for the standard (8,
 * 16, 32, 64) bit widths when they don't have exact representations. It is
 * therefore important that the definitions be kept in order of decending
 * widths.
 *
 * Note that C++ should not check __STDC_LIMIT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#ifdef __INT64_TYPE__
#define INT64_MAX INT64_C(9223372036854775807)
#define INT64_MIN (-INT64_C(9223372036854775807) - 1)
#define UINT64_MAX UINT64_C(18446744073709551615)
#define __INT_LEAST64_MIN INT64_MIN
#define __INT_LEAST64_MAX INT64_MAX
#define __UINT_LEAST64_MAX UINT64_MAX
#endif /* __INT64_TYPE__ */

#ifdef __INT_LEAST64_MIN
#define INT_LEAST64_MIN __INT_LEAST64_MIN
#define INT_LEAST64_MAX __INT_LEAST64_MAX
#define UINT_LEAST64_MAX __UINT_LEAST64_MAX
#define INT_FAST64_MIN __INT_LEAST64_MIN
#define INT_FAST64_MAX __INT_LEAST64_MAX
#define UINT_FAST64_MAX __UINT_LEAST64_MAX
#endif /* __INT_LEAST64_MIN */

#ifdef __INT32_TYPE__
#define INT32_MAX INT32_C(2147483647)
#define INT32_MIN (-INT32_C(2147483647) - 1)
#define UINT32_MAX UINT32_C(4294967295)
#define __INT_LEAST32_MIN INT32_MIN
#define __INT_LEAST32_MAX INT32_MAX
#define __UINT_LEAST32_MAX UINT32_MAX
#endif /* __INT32_TYPE__ */

#ifdef __INT_LEAST32_MIN
#define INT_LEAST32_MIN __INT_LEAST32_MIN
#define INT_LEAST32_MAX __INT_LEAST32_MAX
#define UINT_LEAST32_MAX __UINT_LEAST32_MAX
#define INT_FAST32_MIN __INT_LEAST32_MIN
#define INT_FAST32_MAX __INT_LEAST32_MAX
#define UINT_FAST32_MAX __UINT_LEAST32_MAX
#endif /* __INT_LEAST32_MIN */

#ifdef __INT16_TYPE__
#define INT16_MAX INT16_C(32767)
#define INT16_MIN (-INT16_C(32767) - 1)
#define UINT16_MAX UINT16_C(65535)
#define __INT_LEAST16_MIN INT16_MIN
#define __INT_LEAST16_MAX INT16_MAX
#define __UINT_LEAST16_MAX UINT16_MAX
#endif /* __INT16_TYPE__ */

#ifdef __INT_LEAST16_MIN
#define INT_LEAST16_MIN __INT_LEAST16_MIN
#define INT_LEAST16_MAX __INT_LEAST16_MAX
#define UINT_LEAST16_MAX __UINT_LEAST16_MAX
#define INT_FAST16_MIN __INT_LEAST16_MIN
#define INT_FAST16_MAX __INT_LEAST16_MAX
#define UINT_FAST16_MAX __UINT_LEAST16_MAX
#endif /* __INT_LEAST16_MIN */


#ifdef __INT8_TYPE__
#define INT8_MAX INT8_C(127)
#define INT8_MIN (-INT8_C(127) - 1)
#define UINT8_MAX UINT8_C(255)
#define __INT_LEAST8_MIN INT8_MIN
#define __INT_LEAST8_MAX INT8_MAX
#define __UINT_LEAST8_MAX UINT8_MAX
#endif /* __INT8_TYPE__ */

#ifdef __INT_LEAST8_MIN
#define INT_LEAST8_MIN __INT_LEAST8_MIN
#define INT_LEAST8_MAX __INT_LEAST8_MAX
#define UINT_LEAST8_MAX __UINT_LEAST8_MAX
#define INT_FAST8_MIN __INT_LEAST8_MIN
#define INT_FAST8_MAX __INT_LEAST8_MAX
#define UINT_FAST8_MAX __UINT_LEAST8_MAX
#endif /* __INT_LEAST8_MIN */

/* Some utility macros */
#define __INTN_MIN(n) __stdint_join3(INT, n, _MIN)
#define __INTN_MAX(n) __stdint_join3(INT, n, _MAX)
#define __UINTN_MAX(n) __stdint_join3(UINT, n, _MAX)
#define __INTN_C(n, v) __stdint_join3(INT, n, _C(v))
#define __UINTN_C(n, v) __stdint_join3(UINT, n, _C(v))

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#define INTPTR_MIN (-__INTPTR_MAX__ - 1)
#define INTPTR_MAX __INTPTR_MAX__
#define UINTPTR_MAX __UINTPTR_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__ - 1)
#define PTRDIFF_MAX __PTRDIFF_MAX__
#define SIZE_MAX __SIZE_MAX__

/* ISO9899:2011 7.20 (C11 Annex K): Define RSIZE_MAX if __STDC_WANT_LIB_EXT1__
 * is enabled. */
#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define RSIZE_MAX (SIZE_MAX >> 1)
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define INTMAX_MIN (-__INTMAX_MAX__ - 1)
#define INTMAX_MAX __INTMAX_MAX__
#define UINTMAX_MAX __UINTMAX_MAX__

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN __INTN_MIN(__SIG_ATOMIC_WIDTH__)
#define SIG_ATOMIC_MAX __INTN_MAX(__SIG_ATOMIC_WIDTH__)
#ifdef __WINT_UNSIGNED__
#define WINT_MIN __UINTN_C(__WINT_WIDTH__, 0)
#define WINT_MAX __UINTN_MAX(__WINT_WIDTH__)
#else
#define WINT_MIN __INTN_MIN(__WINT_WIDTH__)
#define WINT_MAX __INTN_MAX(__WINT_WIDTH__)
#endif

#ifndef WCHAR_MAX
#define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
#if __WCHAR_MAX__ == __INTN_MAX(__WCHAR_WIDTH__)
#define WCHAR_MIN __INTN_MIN(__WCHAR_WIDTH__)
#else
#define WCHAR_MIN __UINTN_C(__WCHAR_WIDTH__, 0)
#endif
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define INTMAX_C(v) __int_c(v, __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __int_c(v, __UINTMAX_C_SUFFIX__)


#endif // SLANG_LLVM_H

#else // SLANG_LLVM
#if SLANG_GCC_FAMILY && __GNUC__ < 6
#include <cmath>
#define SLANG_PRELUDE_STD std::
#else
#include <math.h>
#define SLANG_PRELUDE_STD
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#endif // SLANG_LLVM

// Is intptr_t not equal to equal-width sized integer type?
#if defined(__APPLE__)
#define SLANG_INTPTR_TYPE_IS_DISTINCT 1
#else
#define SLANG_INTPTR_TYPE_IS_DISTINCT 0
#endif

#if defined(_MSC_VER)
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
// #   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport))
// __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
#define SLANG_PRELUDE_EXTERN_C extern "C"
#define SLANG_PRELUDE_EXTERN_C_START \
    extern "C"                       \
    {
#define SLANG_PRELUDE_EXTERN_C_END }
#else
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END
#endif

#define SLANG_PRELUDE_EXPORT SLANG_PRELUDE_EXTERN_C SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_START SLANG_PRELUDE_EXTERN_C_START SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_END SLANG_PRELUDE_EXTERN_C_END

#ifndef INFINITY
// Must overflow for double
#define INFINITY float(1e+300 * 1e+300)
#endif

#ifndef SLANG_INFINITY
#define SLANG_INFINITY INFINITY
#endif

// Detect the compiler type

#ifndef SLANG_COMPILER
#define SLANG_COMPILER

/*
Compiler defines, see http://sourceforge.net/p/predef/wiki/Compilers/
NOTE that SLANG_VC holds the compiler version - not just 1 or 0
*/
#if defined(_MSC_VER)
#if _MSC_VER >= 1900
#define SLANG_VC 14
#elif _MSC_VER >= 1800
#define SLANG_VC 12
#elif _MSC_VER >= 1700
#define SLANG_VC 11
#elif _MSC_VER >= 1600
#define SLANG_VC 10
#elif _MSC_VER >= 1500
#define SLANG_VC 9
#else
#error "unknown version of Visual C++ compiler"
#endif
#elif defined(__clang__)
#define SLANG_CLANG 1
#elif defined(__SNC__)
#define SLANG_SNC 1
#elif defined(__ghs__)
#define SLANG_GHS 1
#elif defined(__GNUC__) /* note: __clang__, __SNC__, or __ghs__ imply __GNUC__ */
#define SLANG_GCC 1
#else
#error "unknown compiler"
#endif
/*
Any compilers not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_VC
#define SLANG_VC 0
#endif
#ifndef SLANG_CLANG
#define SLANG_CLANG 0
#endif
#ifndef SLANG_SNC
#define SLANG_SNC 0
#endif
#ifndef SLANG_GHS
#define SLANG_GHS 0
#endif
#ifndef SLANG_GCC
#define SLANG_GCC 0
#endif
#endif /* SLANG_COMPILER */

/*
The following section attempts to detect the target platform being compiled for.

If an application defines `SLANG_PLATFORM` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_PLATFORM
#define SLANG_PLATFORM
/**
Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
*/
#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_PARTITION_APP
#define SLANG_WINRT 1 /* Windows Runtime, either on Windows RT or Windows 8 */
#elif defined(XBOXONE)
#define SLANG_XBOXONE 1
#elif defined(_WIN64) /* note: XBOXONE implies _WIN64 */
#define SLANG_WIN64 1
#elif defined(_M_PPC)
#define SLANG_X360 1
#elif defined(_WIN32) /* note: _M_PPC implies _WIN32 */
#define SLANG_WIN32 1
#elif defined(__ANDROID__)
#define SLANG_ANDROID 1
#elif defined(__linux__) || defined(__CYGWIN__) /* note: __ANDROID__ implies __linux__ */
#define SLANG_LINUX 1
#elif defined(__APPLE__) && !defined(SLANG_LLVM)
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#define SLANG_OSX 1
#else
#define SLANG_IOS 1
#endif
#elif defined(__APPLE__)
// On `slang-llvm` we can't inclue "TargetConditionals.h" in general, so for now assume its
// OSX.
#define SLANG_OSX 1
#elif defined(__CELLOS_LV2__)
#define SLANG_PS3 1
#elif defined(__ORBIS__)
#define SLANG_PS4 1
#elif defined(__SNC__) && defined(__arm__)
#define SLANG_PSP2 1
#elif defined(__ghs__)
#define SLANG_WIIU 1
#else
#error "unknown target platform"
#endif


/*
Any platforms not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_WINRT
#define SLANG_WINRT 0
#endif
#ifndef SLANG_XBOXONE
#define SLANG_XBOXONE 0
#endif
#ifndef SLANG_WIN64
#define SLANG_WIN64 0
#endif
#ifndef SLANG_X360
#define SLANG_X360 0
#endif
#ifndef SLANG_WIN32
#define SLANG_WIN32 0
#endif
#ifndef SLANG_ANDROID
#define SLANG_ANDROID 0
#endif
#ifndef SLANG_LINUX
#define SLANG_LINUX 0
#endif
#ifndef SLANG_IOS
#define SLANG_IOS 0
#endif
#ifndef SLANG_OSX
#define SLANG_OSX 0
#endif
#ifndef SLANG_PS3
#define SLANG_PS3 0
#endif
#ifndef SLANG_PS4
#define SLANG_PS4 0
#endif
#ifndef SLANG_PSP2
#define SLANG_PSP2 0
#endif
#ifndef SLANG_WIIU
#define SLANG_WIIU 0
#endif
#endif /* SLANG_PLATFORM */

/* Shorthands for "families" of compilers/platforms */
#define SLANG_GCC_FAMILY (SLANG_CLANG || SLANG_SNC || SLANG_GHS || SLANG_GCC)
#define SLANG_WINDOWS_FAMILY (SLANG_WINRT || SLANG_WIN32 || SLANG_WIN64)
#define SLANG_MICROSOFT_FAMILY (SLANG_XBOXONE || SLANG_X360 || SLANG_WINDOWS_FAMILY)
#define SLANG_LINUX_FAMILY (SLANG_LINUX || SLANG_ANDROID)
#define SLANG_APPLE_FAMILY (SLANG_IOS || SLANG_OSX) /* equivalent to #if __APPLE__ */
#define SLANG_UNIX_FAMILY \
    (SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY) /* shortcut for unix/posix platforms */

// GCC Specific
#if SLANG_GCC_FAMILY

#if INTPTR_MAX == INT64_MAX
#define SLANG_64BIT 1
#else
#define SLANG_64BIT 0
#endif

#define SLANG_BREAKPOINT(id) __builtin_trap()

// Use this macro instead of offsetof, because gcc produces warning if offsetof is used on a
// non POD type, even though it produces the correct result
#define SLANG_OFFSET_OF(T, ELEMENT) (size_t(&((T*)1)->ELEMENT) - 1)
#endif // SLANG_GCC_FAMILY

// Microsoft VC specific
#if SLANG_VC

#define SLANG_BREAKPOINT(id) __debugbreak();

#endif // SLANG_VC

// Default impls

#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(X, Y) offsetof(X, Y)
#endif

#ifndef SLANG_BREAKPOINT
// Make it crash with a write to 0!
#define SLANG_BREAKPOINT(id) (*((int*)0) = int(id));
#endif

// If slang.h has been included we don't need any of these definitions
#ifndef SLANG_H

/* Macro for declaring if a method is no throw. Should be set before the return parameter. */
#ifndef SLANG_NO_THROW
#if SLANG_WINDOWS_FAMILY && !defined(SLANG_DISABLE_EXCEPTIONS)
#define SLANG_NO_THROW __declspec(nothrow)
#endif
#endif
#ifndef SLANG_NO_THROW
#define SLANG_NO_THROW
#endif

/* The `SLANG_STDCALL` and `SLANG_MCALL` defines are used to set the calling
convention for interface methods.
*/
#ifndef SLANG_STDCALL
#if SLANG_MICROSOFT_FAMILY
#define SLANG_STDCALL __stdcall
#else
#define SLANG_STDCALL
#endif
#endif
#ifndef SLANG_MCALL
#define SLANG_MCALL SLANG_STDCALL
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

// TODO(JS): Should these be in slang-cpp-types.h?
// They are more likely to clash with slang.h

struct SlangUUID
{
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};

typedef int32_t SlangResult;

struct ISlangUnknown
{
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() = 0;
};

#define SLANG_COM_INTERFACE(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)             \
public:                                                                          \
    SLANG_FORCE_INLINE static const SlangUUID& getTypeGuid()                     \
    {                                                                            \
        static const SlangUUID guid = {a, b, c, d0, d1, d2, d3, d4, d5, d6, d7}; \
        return guid;                                                             \
    }
#endif // SLANG_H

// Includes

#ifndef SLANG_PRELUDE_SCALAR_INTRINSICS_H
#define SLANG_PRELUDE_SCALAR_INTRINSICS_H

#if !defined(SLANG_LLVM) && SLANG_PROCESSOR_X86_64 && SLANG_VC
//  If we have visual studio and 64 bit processor, we can assume we have popcnt, and can include
//  x86 intrinsics
#include <intrin.h>
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_PRELUDE_PI
#define SLANG_PRELUDE_PI 3.14159265358979323846
#endif

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

// 32 bit cast conversions
SLANG_FORCE_INLINE int32_t _bitCastFloatToInt(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}
SLANG_FORCE_INLINE float _bitCastIntToFloat(int32_t i)
{
    Union32 u;
    u.i = i;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t _bitCastFloatToUInt(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE float _bitCastUIntToFloat(uint32_t ui)
{
    Union32 u;
    u.u = ui;
    return u.f;
}

// ----------------------------- F32 -----------------------------------------

// Helpers
SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
float F32_ceil(float f);
float F32_floor(float f);
float F32_round(float f);
float F32_sin(float f);
float F32_cos(float f);
float F32_tan(float f);
float F32_asin(float f);
float F32_acos(float f);
float F32_atan(float f);
float F32_sinh(float f);
float F32_cosh(float f);
float F32_tanh(float f);
float F32_asinh(float f);
float F32_acosh(float f);
float F32_atanh(float f);
float F32_log2(float f);
float F32_log(float f);
float F32_log10(float f);
float F32_exp2(float f);
float F32_exp(float f);
float F32_abs(float f);
float F32_trunc(float f);
float F32_sqrt(float f);

bool F32_isnan(float f);
bool F32_isfinite(float f);
bool F32_isinf(float f);

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return a > b ? a : b;
}
float F32_pow(float a, float b);
float F32_fmod(float a, float b);
float F32_remainder(float a, float b);
float F32_atan2(float a, float b);

float F32_frexp(float x, int* e);

float F32_modf(float x, float* ip);

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else

// Unary
SLANG_FORCE_INLINE float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE float F32_asinh(float f)
{
    return ::asinhf(f);
}
SLANG_FORCE_INLINE float F32_acosh(float f)
{
    return ::acoshf(f);
}
SLANG_FORCE_INLINE float F32_atanh(float f)
{
    return ::atanhf(f);
}
SLANG_FORCE_INLINE float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE float F32_sqrt(float f)
{
    return ::sqrtf(f);
}

SLANG_FORCE_INLINE bool F32_isnan(float f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F32_isfinite(float f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F32_isinf(float f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE float F32_frexp(float x, int* e)
{
    return ::frexpf(x, e);
}

SLANG_FORCE_INLINE float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}

#endif

SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = radians * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return (a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE float F32_rsqrt(float f)
{
    return 1.0f / F32_sqrt(f);
}
SLANG_FORCE_INLINE int F32_sign(float f)
{
    return (f == 0.0f) ? 0 : ((f < 0.0f) ? -1 : 1);
}
SLANG_FORCE_INLINE float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// ----------------------------- F64 -----------------------------------------

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
double F64_ceil(double f);
double F64_floor(double f);
double F64_round(double f);
double F64_sin(double f);
double F64_cos(double f);
double F64_tan(double f);
double F64_asin(double f);
double F64_acos(double f);
double F64_atan(double f);
double F64_sinh(double f);
double F64_cosh(double f);
double F64_tanh(double f);
double F64_asinh(double f);
double F64_acosh(double f);
double F64_atanh(double f);
double F64_log2(double f);
double F64_log(double f);
double F64_log10(double f);
double F64_exp2(double f);
double F64_exp(double f);
double F64_abs(double f);
double F64_trunc(double f);
double F64_sqrt(double f);

bool F64_isnan(double f);
bool F64_isfinite(double f);
bool F64_isinf(double f);

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return a > b ? a : b;
}
double F64_pow(double a, double b);
double F64_fmod(double a, double b);
double F64_remainder(double a, double b);
double F64_atan2(double a, double b);

double F64_frexp(double x, int* e);

double F64_modf(double x, double* ip);

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else // SLANG_LLVM

// Unary
SLANG_FORCE_INLINE double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE double F64_sqrt(double f)
{
    return ::sqrt(f);
}


SLANG_FORCE_INLINE bool F64_isnan(double f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F64_isfinite(double f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F64_isinf(double f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

#endif // SLANG_LLVM

SLANG_FORCE_INLINE double F64_rsqrt(double f)
{
    return 1.0 / F64_sqrt(f);
}
SLANG_FORCE_INLINE int F64_sign(double f)
{
    return (f == 0.0) ? 0 : ((f < 0.0) ? -1 : 1);
}
SLANG_FORCE_INLINE double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    double a = radians * (1.0f / (SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F64_floor(a);
    // Convert back to 0 - 2pi range
    return (a * (SLANG_PRELUDE_PI * 2));
}

// ----------------------------- F16 -----------------------------------------

// This impl is based on FloatToHalf that is in Slang codebase
SLANG_FORCE_INLINE uint32_t f32tof16(const float value)
{
    const uint32_t inBits = _bitCastFloatToUInt(value);

    // bits initially set to just the sign bit
    uint32_t bits = (inBits >> 16) & 0x8000;
    // Mantissa can't be used as is, as it holds last bit, for rounding.
    uint32_t m = (inBits >> 12) & 0x07ff;
    uint32_t e = (inBits >> 23) & 0xff;

    if (e < 103)
    {
        // It's zero
        return bits;
    }
    if (e == 0xff)
    {
        // Could be a NAN or INF. Is INF if *input* mantissa is 0.

        // Remove last bit for rounding to make output mantissa.
        m >>= 1;

        // We *assume* float16/float32 signaling bit and remaining bits
        // semantics are the same. (The signalling bit convention is target specific!).
        // Non signal bit's usage within mantissa for a NAN are also target specific.

        // If the m is 0, it could be because the result is INF, but it could also be because all
        // the bits that made NAN were dropped as we have less mantissa bits in f16.

        // To fix for this we make non zero if m is 0 and the input mantissa was not.
        // This will (typically) produce a signalling NAN.
        m += uint32_t(m == 0 && (inBits & 0x007fffffu));

        // Combine for output
        return (bits | 0x7c00u | m);
    }
    if (e > 142)
    {
        // INF.
        return bits | 0x7c00u;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

static const float g_f16tof32Magic = _bitCastIntToFloat((127 + (127 - 15)) << 23);

SLANG_FORCE_INLINE float f16tof32(const uint32_t value)
{
    const uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = (value & 0x03ff);

    if (exponent == 0)
    {
        // If mantissa is 0 we are done, as output is 0.
        // If it's not zero we must have a denormal.
        if (mantissa)
        {
            // We have a denormal so use the magic to do exponent adjust
            return _bitCastIntToFloat(sign | ((value & 0x7fff) << 13)) * g_f16tof32Magic;
        }
    }
    else
    {
        // If the exponent is NAN or INF exponent is 0x1f on input.
        // If that's the case, we just need to set the exponent to 0xff on output
        // and the mantissa can just stay the same. If its 0 it's INF, else it is NAN and we just
        // copy the bits
        //
        // Else we need to correct the exponent in the normalized case.
        exponent = (exponent == 0x1F) ? 0xff : (exponent + (-15 + 127));
    }

    return _bitCastUIntToFloat(sign | (exponent << 23) | (mantissa << 13));
}

#ifndef SLANG_LLVM
#if __cplusplus >= 202302L
#include <stdfloat> // C++23
#else
// Define __STDC_WANT_IEC_60559_TYPES_EXT__ for compilers with reliable _Float16 support:
// - Clang 15+
// - GCC 12+
#if (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#endif
#include <float.h>
#endif // __STDC_WANT_IEC_60559_TYPES_EXT__
#endif // (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#endif // C++23

#ifdef FLT16_MIN
typedef _Float16 half;
#elif __STDCPP_FLOAT16_T__ == 1
typedef std::float16_t half;
#else
uint32_t f32tof16(const float value);
float f16tof32(const uint32_t value);
struct half
{
    uint16_t data;

    half() = default;
    explicit half(float f) { store(f); }

    SLANG_FORCE_INLINE void store(float f) { data = f32tof16(f); }
    SLANG_FORCE_INLINE float load() const { return f16tof32(data); }

    half operator+(half other) const { return half(load() + other.load()); }
    half operator-(half other) const { return half(load() - other.load()); }
    half operator*(half other) const { return half(load() * other.load()); }
    half operator/(half other) const { return half(load() / other.load()); }
    half& operator+=(half other)
    {
        store(load() + other.load());
        return *this;
    }
    half& operator-=(half other)
    {
        store(load() - other.load());
        return *this;
    }
    half& operator*=(half other)
    {
        store(load() * other.load());
        return *this;
    }
    half& operator/=(half other)
    {
        store(load() / other.load());
        return *this;
    }

    bool operator<(half other) const { return load() < other.load(); }
    bool operator>(half other) const { return load() > other.load(); }
    bool operator<=(half other) const { return load() <= other.load(); }
    bool operator>=(half other) const { return load() >= other.load(); }
    bool operator==(half other) const { return load() == other.load(); }
    bool operator!=(half other) const { return load() != other.load(); }

    explicit operator float() const { return load(); }
};
#endif

half U16_ashalf(uint16_t x);

union Union16
{
    uint16_t u;
    int16_t i;
    half h;
};

SLANG_FORCE_INLINE uint16_t F16_asuint(half h)
{
    Union16 u;
    u.h = h;
    return u.u;
}

SLANG_FORCE_INLINE int16_t F16_asint(half h)
{
    Union16 u;
    u.h = h;
    return u.i;
}

SLANG_FORCE_INLINE half F16_ceil(half f)
{
    return half(F32_ceil(float(f)));
}

SLANG_FORCE_INLINE half F16_floor(half f)
{
    return half(F32_floor(float(f)));
}

SLANG_FORCE_INLINE half F16_round(half f)
{
    return half(F32_round(float(f)));
}

SLANG_FORCE_INLINE half F16_sin(half f)
{
    return half(F32_sin(float(f)));
}

SLANG_FORCE_INLINE half F16_cos(half f)
{
    return half(F32_cos(float(f)));
}

SLANG_FORCE_INLINE half F16_tan(half f)
{
    return half(F32_tan(float(f)));
}

SLANG_FORCE_INLINE half F16_asin(half f)
{
    return half(F32_asin(float(f)));
}

SLANG_FORCE_INLINE half F16_acos(half f)
{
    return half(F32_acos(float(f)));
}

SLANG_FORCE_INLINE half F16_atan(half f)
{
    return half(F32_atan(float(f)));
}

SLANG_FORCE_INLINE half F16_sinh(half f)
{
    return half(F32_sinh(float(f)));
}

SLANG_FORCE_INLINE half F16_cosh(half f)
{
    return half(F32_cosh(float(f)));
}

SLANG_FORCE_INLINE half F16_tanh(half f)
{
    return half(F32_tanh(float(f)));
}

SLANG_FORCE_INLINE half F16_asinh(half f)
{
    return half(F32_asinh(float(f)));
}

SLANG_FORCE_INLINE half F16_acosh(half f)
{
    return half(F32_acosh(float(f)));
}

SLANG_FORCE_INLINE half F16_atanh(half f)
{
    return half(F32_atanh(float(f)));
}

SLANG_FORCE_INLINE half F16_log2(half f)
{
    return half(F32_log2(float(f)));
}

SLANG_FORCE_INLINE half F16_log(half f)
{
    return half(F32_log(float(f)));
}

SLANG_FORCE_INLINE half F16_log10(half f)
{
    return half(F32_log10(float(f)));
}

SLANG_FORCE_INLINE half F16_exp2(half f)
{
    return half(F32_exp2(float(f)));
}

SLANG_FORCE_INLINE half F16_exp(half f)
{
    return half(F32_exp(float(f)));
}

SLANG_FORCE_INLINE half F16_abs(half f)
{
    return U16_ashalf(F16_asuint(f) & 0x7FFF);
}

SLANG_FORCE_INLINE half F16_trunc(half f)
{
    return half(F32_trunc(float(f)));
}

SLANG_FORCE_INLINE half F16_sqrt(half f)
{
    return half(F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE bool F16_isnan(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) != 0;
}

SLANG_FORCE_INLINE bool F16_isfinite(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) != 0x7C00;
}

SLANG_FORCE_INLINE bool F16_isinf(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) == 0;
}

SLANG_FORCE_INLINE half F16_min(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a < b ? a : b;
}

SLANG_FORCE_INLINE half F16_max(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a > b ? a : b;
}

SLANG_FORCE_INLINE half F16_pow(half a, half b)
{
    return half(F32_pow(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_fmod(half a, half b)
{
    return half(F32_fmod(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_remainder(half a, half b)
{
    return half(F32_remainder(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_atan2(half a, half b)
{
    return half(F32_atan2(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_frexp(half x, int* e)
{
    return half(F32_frexp(float(x), e));
}

SLANG_FORCE_INLINE half F16_modf(half x, half* ip)
{
    float ipf;
    float res = F32_modf(float(x), &ipf);
    *ip = half(ipf);
    return half(res);
}

SLANG_FORCE_INLINE half F16_fma(half a, half b, half c)
{
    return half(F32_fma(float(a), float(b), float(c)));
}

SLANG_FORCE_INLINE half F16_calcSafeRadians(half radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = float(radians) * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return half(a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE half F16_rsqrt(half f)
{
    return half(1.0f / F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE int F16_sign(half f)
{
    uint16_t u = F16_asuint(f);
    if ((u & 0x7FFF) == 0)
        return 0;
    return (u & 0x8000) != 0 ? -1 : 1;
}

SLANG_FORCE_INLINE half F16_frac(half h)
{
    float f = float(h);
    return half(f - F32_floor(f));
}

// ----------------------------- U16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U16_countbits(uint16_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(uint32_t(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt16(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE half U16_ashalf(uint16_t x)
{
    Union16 u;
    u.u = x;
    return u.h;
}

// ----------------------------- I16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit __popcnt yet, just cast and use 16bit variant
    return U16_countbits(uint16_t(v));
}

// ----------------------------- I8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I8_countbits(int16_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U32 -----------------------------------------

SLANG_FORCE_INLINE uint32_t U32_abs(uint32_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE uint32_t U32_countbits(uint32_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbitlow(uint32_t v)
{
    if (v == 0)
        return ~0u;

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward(&index, v) ? index : ~0u;
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 32 && !(v & (1u << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbithigh(uint32_t v)
{
    if (v == 0)
        return ~0u;
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 31 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse(&index, v) ? index : ~0u;
#else
    // Generic implementation - find highest set bit
    int result = 31;
    while (result >= 0 && !(v & (1u << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_reversebits(uint32_t v)
{
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
    v = (v >> 16) | (v << 16);
    return v;
}

// ----------------------------- I32 -----------------------------------------

SLANG_FORCE_INLINE int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbitlow(int32_t v)
{
    return U32_firstbitlow(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbithigh(int32_t v)
{
    if (v < 0)
        v = ~v;
    return U32_firstbithigh(uint32_t(v));
}

SLANG_FORCE_INLINE int32_t I32_reversebits(int32_t v)
{
    return U32_reversebits(int32_t(v));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE uint64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t U64_countbits(uint64_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return uint32_t(__builtin_popcountll(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return uint32_t(__popcnt64(v));
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbitlow(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 64 && !(v & (uint64_t(1) << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbithigh(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 63 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find highest set bit
    int result = 63;
    while (result >= 0 && !(v & (uint64_t(1) << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint64_t U64_reversebits(uint64_t v)
{
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    v = ((v >> 8) & 0x00FF00FF00FF00FFull) | ((v & 0x00FF00FF00FF00FFull) << 8);
    v = ((v >> 16) & 0x0000FFFF0000FFFFull) | ((v & 0x0000FFFF0000FFFFull) << 16);
    v = (v >> 32) | (v << 32);
    return v;
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbitlow(int64_t v)
{
    return U64_firstbitlow(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbithigh(int64_t v)
{
    if (v < 0)
        v = ~v;
    return U64_firstbithigh(uint64_t(v));
}

SLANG_FORCE_INLINE int64_t I64_reversebits(int64_t v)
{
    return int64_t(U64_reversebits(uint64_t(v)));
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- Interlocked ---------------------------------

#if SLANG_LLVM

#else // SLANG_LLVM

#ifdef _WIN32
#include <intrin.h>
#endif

SLANG_FORCE_INLINE void InterlockedAdd(uint32_t* dest, uint32_t value, uint32_t* oldValue)
{
#ifdef _WIN32
    *oldValue = _InterlockedExchangeAdd((long*)dest, (long)value);
#else
    *oldValue = __sync_fetch_and_add(dest, value);
#endif
}

#endif // SLANG_LLVM


// ----------------------- fmod --------------------------
SLANG_FORCE_INLINE float _slang_fmod(float x, float y)
{
    return F32_fmod(x, y);
}
SLANG_FORCE_INLINE double _slang_fmod(double x, double y)
{
    return F64_fmod(x, y);
}

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_H
#define SLANG_PRELUDE_CPP_TYPES_H

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#else
#define SLANG_PRELUDE_ASSERT(VALUE)
#endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++
 * code.
 */

template<typename T, int COUNT>
struct Vector;

template<typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar) { x = scalar; }
    template<typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar) { x = y = scalar; }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template<typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar) { x = y = z = scalar; }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template<typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar) { x = y = z = w = scalar; }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T, int N>
SLANG_FORCE_INLINE Vector<T, N> _slang_select(
    Vector<bool, N> condition,
    Vector<T, N> v0,
    Vector<T, N> v1)
{
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = condition[i] ? v0[i] : v1[i];
    }
    return result;
}

template<typename T>
SLANG_FORCE_INLINE T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T, N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op)            \
    template<int n>                              \
    SLANG_FORCE_INLINE Vector<T, n> operator op( \
        const Vector<T, n>& thisVal,             \
        const Vector<T, n>& other)               \
    {                                            \
        Vector<T, n> result;                     \
        for (int i = 0; i < n; i++)              \
            result[i] = thisVal[i] op other[i];  \
        return result;                           \
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op)       \
    template<int n>                                 \
    SLANG_FORCE_INLINE Vector<bool, n> operator op( \
        const Vector<T, n>& thisVal,                \
        const Vector<T, n>& other)                  \
    {                                               \
        Vector<bool, n> result;                     \
        for (int i = 0; i < n; i++)                 \
            result[i] = thisVal[i] op other[i];     \
        return result;                              \
    }

#define SLANG_VECTOR_UNARY_OP(T, op)                                         \
    template<int n>                                                          \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    {                                                                        \
        Vector<T, n> result;                                                 \
        for (int i = 0; i < n; i++)                                          \
            result[i] = op thisVal[i];                                       \
        return result;                                                       \
    }
#define SLANG_INT_VECTOR_OPS(T)           \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_BINARY_OP(T, &)          \
    SLANG_VECTOR_BINARY_OP(T, |)          \
    SLANG_VECTOR_BINARY_OP(T, &&)         \
    SLANG_VECTOR_BINARY_OP(T, ||)         \
    SLANG_VECTOR_BINARY_OP(T, ^)          \
    SLANG_VECTOR_BINARY_OP(T, %)          \
    SLANG_VECTOR_BINARY_OP(T, >>)         \
    SLANG_VECTOR_BINARY_OP(T, <<)         \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=) \
    SLANG_VECTOR_UNARY_OP(T, !)           \
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T)         \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_UNARY_OP(T, -)           \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_VECTOR_OPS(intptr_t)
SLANG_INT_VECTOR_OPS(uintptr_t)
#endif

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T)                      \
    template<int N>                                     \
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    {                                                   \
        Vector<T, N> result;                            \
        for (int i = 0; i < N; i++)                     \
            result[i] = 0 - thisVal[i];                 \
        return result;                                  \
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_VECTOR_INT_NEG_OP(intptr_t)
SLANG_VECTOR_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_VECTOR_MOD(T)                                               \
    template<int N>                                                             \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {                                                                           \
        Vector<T, N> result;                                                    \
        for (int i = 0; i < N; i++)                                             \
            result[i] = _slang_fmod(left[i], right[i]);                         \
        return result;                                                          \
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    const Vector<T, COLS>& operator[](size_t index) const { return rows[index]; }
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0) { rows[0] = row0; }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(
        const Vector<T, COLS>& row0,
        const Vector<T, COLS>& row1,
        const Vector<T, COLS>& row2,
        const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow)
            minRow = otherRow;
        if (minCol > otherCol)
            minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[1][0] = v2;
        rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
            rows[3][0] = v6;
            rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[1][0] = v3;
        rows[1][1] = v4;
        rows[1][2] = v5;
        rows[2][0] = v6;
        rows[2][1] = v7;
        rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
            rows[2][0] = v8;
            rows[2][1] = v9;
            rows[2][2] = v10;
            rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
            rows[2][0] = v6;
            rows[2][1] = v7;
            rows[2][2] = v8;
            rows[3][0] = v9;
            rows[3][1] = v10;
            rows[3][2] = v11;
        }
    }
    Matrix(
        T v0,
        T v1,
        T v2,
        T v3,
        T v4,
        T v5,
        T v6,
        T v7,
        T v8,
        T v9,
        T v10,
        T v11,
        T v12,
        T v13,
        T v14,
        T v15)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[0][3] = v3;
        rows[1][0] = v4;
        rows[1][1] = v5;
        rows[1][2] = v6;
        rows[1][3] = v7;
        rows[2][0] = v8;
        rows[2][1] = v9;
        rows[2][2] = v10;
        rows[2][3] = v11;
        rows[3][0] = v12;
        rows[3][1] = v13;
        rows[3][2] = v14;
        rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op)                                                         \
    template<int R, int C>                                                                    \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                   \
        return result;                                                                        \
    }

#define SLANG_MATRIX_BINARY_COMPARE_OP(T, op)                                                    \
    template<int R, int C>                                                                       \
    Matrix<bool, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                            \
        Matrix<bool, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                              \
            for (int j = 0; j < C; j++)                                                          \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                      \
        return result;                                                                           \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                            \
    template<int R, int C>                                      \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                           \
        Matrix<T, R, C> result;                                 \
        for (int i = 0; i < R; i++)                             \
            for (int j = 0; j < C; j++)                         \
                result[i].rows[i][j] = op thisVal.rows[i][j];   \
        return result;                                          \
    }

#define SLANG_INT_MATRIX_OPS(T)           \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_BINARY_OP(T, &)          \
    SLANG_MATRIX_BINARY_OP(T, |)          \
    SLANG_MATRIX_BINARY_OP(T, &&)         \
    SLANG_MATRIX_BINARY_OP(T, ||)         \
    SLANG_MATRIX_BINARY_OP(T, ^)          \
    SLANG_MATRIX_BINARY_OP(T, %)          \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=) \
    SLANG_MATRIX_UNARY_OP(T, !)           \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T)         \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_UNARY_OP(T, -)           \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_MATRIX_OPS(intptr_t)
SLANG_INT_MATRIX_OPS(uintptr_t)
#endif

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T)                                        \
    template<int R, int C>                                                \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                     \
        Matrix<T, R, C> result;                                           \
        for (int i = 0; i < R; i++)                                       \
            for (int j = 0; j < C; j++)                                   \
                result.rows[i][j] = 0 - thisVal.rows[i][j];               \
        return result;                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(int8_t)
SLANG_MATRIX_INT_NEG_OP(int16_t)
SLANG_MATRIX_INT_NEG_OP(int64_t)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(uint8_t)
SLANG_MATRIX_INT_NEG_OP(uint16_t)
SLANG_MATRIX_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_MATRIX_INT_NEG_OP(intptr_t)
SLANG_MATRIX_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_MATRIX_MOD(T)                                                             \
    template<int R, int C>                                                                    \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]);           \
        return result;                                                                        \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif


typedef Vector<float, 2> float2;
typedef Vector<float, 3> float3;
typedef Vector<float, 4> float4;

typedef Vector<int32_t, 2> int2;
typedef Vector<int32_t, 3> int3;
typedef Vector<int32_t, 4> int4;

typedef Vector<uint32_t, 2> uint2;
typedef Vector<uint32_t, 3> uint3;
typedef Vector<uint32_t, 4> uint4;

// We can just map `NonUniformResourceIndex` type directly to the index type on CPU, as CPU does not
// require any special handling around such accesses.
typedef size_t NonUniformResourceIndex;

// ----------------------------- ResourceType -----------------------------------------

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct RWStructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};

template<typename T>
struct StructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};


template<typename T>
struct RWBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

template<typename T>
struct Buffer
{
    SLANG_FORCE_INLINE const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        *(T*)(((char*)data) + index) = value;
    }

    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

struct ISamplerState;
struct ISamplerComparisonState;

struct SamplerState
{
    ISamplerState* state;
};

struct SamplerComparisonState
{
    ISamplerComparisonState* state;
};

#ifndef SLANG_RESOURCE_SHAPE
#define SLANG_RESOURCE_SHAPE
typedef unsigned int SlangResourceShape;
enum
{
    SLANG_RESOURCE_BASE_SHAPE_MASK = 0x0F,

    SLANG_RESOURCE_NONE = 0x00,

    SLANG_TEXTURE_1D = 0x01,
    SLANG_TEXTURE_2D = 0x02,
    SLANG_TEXTURE_3D = 0x03,
    SLANG_TEXTURE_CUBE = 0x04,
    SLANG_TEXTURE_BUFFER = 0x05,

    SLANG_STRUCTURED_BUFFER = 0x06,
    SLANG_BYTE_ADDRESS_BUFFER = 0x07,
    SLANG_RESOURCE_UNKNOWN = 0x08,
    SLANG_ACCELERATION_STRUCTURE = 0x09,
    SLANG_TEXTURE_SUBPASS = 0x0A,

    SLANG_RESOURCE_EXT_SHAPE_MASK = 0xF0,

    SLANG_TEXTURE_FEEDBACK_FLAG = 0x10,
    SLANG_TEXTURE_ARRAY_FLAG = 0x40,
    SLANG_TEXTURE_MULTISAMPLE_FLAG = 0x80,

    SLANG_TEXTURE_1D_ARRAY = SLANG_TEXTURE_1D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_2D_ARRAY = SLANG_TEXTURE_2D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_CUBE_ARRAY = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

    SLANG_TEXTURE_2D_MULTISAMPLE = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY =
        SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_SUBPASS_MULTISAMPLE = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
};
#endif

//
struct TextureDimensions
{
    void reset()
    {
        shape = 0;
        width = height = depth = 0;
        numberOfLevels = 0;
        arrayElementCount = 0;
    }
    int getDimSizes(uint32_t outDims[4]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        case SLANG_TEXTURE_CUBE:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = 6;
                break;
            }
        }

        if (shape & SLANG_TEXTURE_ARRAY_FLAG)
        {
            outDims[count++] = arrayElementCount;
        }
        return count;
    }
    int getMIPDims(int outDims[3]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_CUBE:
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        }
        return count;
    }
    int calcMaxMIPLevels() const
    {
        int dims[3];
        const int dimCount = getMIPDims(dims);
        for (int count = 1; true; count++)
        {
            bool allOne = true;
            for (int i = 0; i < dimCount; ++i)
            {
                if (dims[i] > 1)
                {
                    allOne = false;
                    dims[i] >>= 1;
                }
            }
            if (allOne)
            {
                return count;
            }
        }
    }

    uint32_t shape;
    uint32_t width, height, depth;
    uint32_t numberOfLevels;
    uint32_t arrayElementCount; ///< For array types, 0 otherwise
};


// Texture

struct ITexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;
    virtual void Load(const int32_t* v, void* outData, size_t dataSize) = 0;
    virtual void Sample(
        SamplerState samplerState,
        const float* loc,
        void* outData,
        size_t dataSize) = 0;
    virtual void SampleLevel(
        SamplerState samplerState,
        const float* loc,
        float level,
        void* outData,
        size_t dataSize) = 0;
};

template<typename T>
struct Texture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, float loc) const
    {
        T out;
        texture->Sample(samplerState, &loc, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, float loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCube
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCubeArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float4& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float4& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! RWTexture !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct IRWTexture : ITexture
{
    /// Get the reference to the element at loc.
    virtual void* refAt(const uint32_t* loc) = 0;
};

template<typename T>
struct RWTexture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int32_t loc) const
    {
        T out;
        texture->Load(&loc, &out, sizeof(out));
        return out;
    }
    T& operator[](uint32_t loc) { return *(T*)texture->refAt(&loc); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint2& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};


template<typename T>
struct RWTexture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int2 loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](uint2 loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

template<typename T>
struct RWTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

// FeedbackTexture

struct FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIN_MIP : FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIP_REGION_USED : FeedbackType
{
};

struct IFeedbackTexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;

    // Note here we pass the optional clamp parameter as a pointer. Passing nullptr means no clamp.
    // This was preferred over having two function definitions, and having to differentiate their
    // names
    virtual void WriteSamplerFeedback(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackBias(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float bias,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackGrad(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* ddx,
        const float* ddy,
        const float* clamp = nullptr) = 0;

    virtual void WriteSamplerFeedbackLevel(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float lod) = 0;
};

template<typename T>
struct FeedbackTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location, float clamp)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy,
        float clamp)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level

    template<typename S>
    void WriteSamplerFeedbackLevel(Texture2D<S> tex, SamplerState samp, float2 location, float lod)
    {
        texture->WriteSamplerFeedbackLevel(tex.texture, samp, &location.x, lod);
    }

    // Without Clamp
    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(Texture2D<S> tex, SamplerState samp, float2 location, float bias)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

template<typename T>
struct FeedbackTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float clamp)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy,
        float clamp)
    {
        texture
            ->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level
    template<typename S>
    void WriteSamplerFeedbackLevel(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float lod)
    {
        texture->WriteSamplerFeedbackLevel(texArray.texture, samp, &location.x, lod);
    }

    // Without Clamp

    template<typename S>
    void WriteSamplerFeedback(Texture2DArray<S> texArray, SamplerState samp, float3 location)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy)
    {
        texture->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

/* Varying input for Compute */

/* Used when running a single thread */
struct ComputeThreadVaryingInput
{
    uint3 groupID;
    uint3 groupThreadID;
};

struct ComputeVaryingInput
{
    uint3 startGroupID; ///< start groupID
    uint3 endGroupID;   ///< Non inclusive end groupID
};

// The uniformEntryPointParams and uniformState must be set to structures that match layout that the
// kernel expects. This can be determined via reflection for example.

typedef void (*ComputeThreadFunc)(
    ComputeThreadVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);
typedef void (*ComputeFunc)(
    ComputeVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif


// Atomic helpers for the CPU target. Needed so kIROp_AtomicAdd and
// friends can lower to native atomic operations on the host — matching
// the semantics of HLSL `InterlockedAdd`, SPIR-V `OpAtomicIAdd`, etc.
// Uses compiler builtins so no <atomic> header is required.
//
// Contract for every `_slang_atomic_add_*` helper below (u32/i32/u64/
// i64): atomically add `val` to `*ptr` and return the PRIOR value (the
// value before the add), matching HLSL `InterlockedAdd` and GLSL
// `atomicAdd`. The 64-bit MSVC variants reinterpret the pointer as
// `volatile long long*` for `_InterlockedExchangeAdd64`, which is sound
// because `sizeof(long long) == 8` (asserted below).
//
// Compiler coverage:
//   - MSVC:            `_InterlockedExchangeAdd`
//   - GCC / Clang:     `__atomic_fetch_add`
//   - SNC / GHS etc.:  falls back to a non-atomic load/add/store.
//     These are console / embedded toolchains not known to ship
//     `__atomic_*` builtins; the fallback is racy under concurrency
//     but keeps the prelude buildable there. Platforms that need
//     real atomics on those compilers can add a bespoke branch.
// === MSVC implementations ===
//
// Use the `_Interlocked*` intrinsics. The 32-bit overload operates on
// `long`; the 64-bit overload operates on `long long`. The static
// asserts above pin the expected widths.
#if SLANG_VC
#include <intrin.h>
static_assert(
    sizeof(long) == 4,
    "_InterlockedExchangeAdd uses `long`; MSVC LLP64 requires sizeof(long)==4");
static_assert(
    sizeof(long long) == 8,
    "_InterlockedExchangeAdd64 uses `long long`; expected 8-byte width");
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    // Returns the PRIOR value, matching HLSL InterlockedAdd and GLSL atomicAdd.
    return static_cast<uint32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return static_cast<int32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return static_cast<uint64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return static_cast<int64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
// === GCC / Clang implementations ===
//
// Use the `__atomic_fetch_add` built-in with relaxed ordering; the
// IR-side AtomicAdd emit ignores the memory-order operand and lets
// each toolchain pick its native default.
#elif SLANG_GCC || SLANG_CLANG
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
// === Non-atomic fallback implementations ===
//
// For compilers without a known atomic builtin (Sony SNC, Green Hills
// MULTI, etc.). Racy under concurrent invocation but keeps the
// prelude compilable; CPU-target coverage on these platforms is
// single-threaded in practice.
#else
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    uint32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    int32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    uint64_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    int64_t old = *ptr;
    *ptr = old + val;
    return old;
}
#endif

// TODO(JS): Hack! Output C++ code from slang can copy uninitialized variables.
#if defined(_MSC_VER)
#pragma warning(disable : 4700)
#endif

#ifndef SLANG_UNROLL
#define SLANG_UNROLL
#endif

#endif

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 20 "/var/folders/ff/b6vctr0n2jjfkdtcm0dt7xdr0000gp/T/sand-dem-cpu-cache/19c5b8536b5e111b21eb1c4e57be4ba2babf0ea551a25b2edbeb4ac3218400e6/solver.slang"
struct Config_0
{
    uint32_t activeTimestep_0;
    uint32_t N_0;
    uint32_t shapes_0;
    uint32_t pairCapacity_0;
    uint32_t contactCapacity_0;
    uint32_t binCapacity_0;
    uint32_t binCount_0;
    Vector<uint32_t, 3>  bins_0;
    uint32_t hashSize_0;
    float dt_0;
    float kn_0;
    float kt_0;
    float mu_0;
    float normalViscosity_0;
    float skin_0;
    float drag_0;
    float maxOverlap_0;
    uint32_t historyStep_0;
    uint32_t meshGeometry_0;
    uint32_t periodic_0;
    uint32_t wallCount_0;
    uint32_t wallBoxEnabled_0;
    uint32_t sourceSurfaceNormal_0;
};


#line 28
struct Resources_0
{
    Config_0 c_0;
    RWStructuredBuffer<uint32_t> control_0;
    RWStructuredBuffer<Vector<float, 4> > cell_0;
    RWStructuredBuffer<Vector<float, 4> > position_0;
    RWStructuredBuffer<Vector<float, 4> > positionRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > velocityRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > momentumRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > rotationRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > referencePositionRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > cellLog_0;
    RWStructuredBuffer<Vector<float, 4> > accumulatedRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > velocity_0;
    RWStructuredBuffer<Vector<float, 4> > rotation_0;
    RWStructuredBuffer<Vector<float, 4> > momentum_0;
    RWStructuredBuffer<Vector<float, 4> > force_0;
    RWStructuredBuffer<Vector<float, 4> > torque_0;
    RWStructuredBuffer<Vector<float, 4> > referencePosition_0;
    RWStructuredBuffer<Vector<float, 4> > referenceCell_0;
    StructuredBuffer<uint32_t> shapeId_0;
    StructuredBuffer<Vector<float, 4> > body_0;
    StructuredBuffer<Vector<float, 4> > invI_0;
    StructuredBuffer<Vector<uint32_t, 4> > grids_0;
    StructuredBuffer<Vector<uint32_t, 4> > sampleInfo_0;
    StructuredBuffer<Vector<float, 4> > gridOrigin_0;
    StructuredBuffer<float> distance_0;
    StructuredBuffer<Vector<float, 4> > sample_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshRoots_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshHullInfo_0;
    StructuredBuffer<Vector<float, 4> > meshPlanes_0;
    StructuredBuffer<Vector<float, 4> > meshMin_0;
    StructuredBuffer<Vector<float, 4> > meshMax_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshLinks_0;
    StructuredBuffer<Vector<float, 4> > meshTriangles_0;
    StructuredBuffer<Vector<float, 4> > meshNormals_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshFrontier_0;
    RWStructuredBuffer<uint32_t> meshPairMask_0;
    RWStructuredBuffer<Vector<float, 4> > meshPairTransform_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshClusterInfo_0;
    StructuredBuffer<Vector<uint32_t, 4> > meshClusterRanges_0;
    StructuredBuffer<Vector<float, 4> > meshClusterMin_0;
    StructuredBuffer<Vector<float, 4> > meshClusterMax_0;
    StructuredBuffer<uint32_t> meshSampleIndices_0;
    RWStructuredBuffer<uint32_t> binCounts_0;
    RWStructuredBuffer<uint32_t> binParticles_0;
    RWStructuredBuffer<uint32_t> pairCounts_0;
    RWStructuredBuffer<uint32_t> pairOffsets_0;
    RWStructuredBuffer<Vector<uint32_t, 2> > pairs_0;
    RWStructuredBuffer<uint32_t> contactCounts_0;
    RWStructuredBuffer<uint32_t> contactOffsets_0;
    RWStructuredBuffer<Vector<uint32_t, 4> > meta_0;
    RWStructuredBuffer<Vector<float, 4> > pointDepth_0;
    RWStructuredBuffer<Vector<float, 4> > normalArea_0;
    RWStructuredBuffer<Vector<float, 4> > slip_0;
    RWStructuredBuffer<Vector<float, 4> > contactForce_0;
    RWStructuredBuffer<float> normalLossPower_0;
    RWStructuredBuffer<Vector<float, 4> > normalDamping_0;
    RWStructuredBuffer<Vector<float, 4> > forceVelocity_0;
    RWStructuredBuffer<Vector<float, 4> > forceOmega_0;
    StructuredBuffer<Vector<uint32_t, 4> > oldMeta_0;
    StructuredBuffer<Vector<float, 4> > oldSlip_0;
    StructuredBuffer<Vector<float, 4> > oldNormal_0;
    StructuredBuffer<uint32_t> oldHash_0;
    RWStructuredBuffer<uint32_t> hash_0;
    RWStructuredBuffer<uint32_t> heads_0;
    RWStructuredBuffer<uint32_t> next_0;
    RWStructuredBuffer<Vector<float, 4> > particleStress_0;
    RWStructuredBuffer<Vector<float, 4> > timestepRows_0;
    RWStructuredBuffer<Vector<float, 4> > particleDiagnostic_0;
    RWStructuredBuffer<Vector<float, 4> > diagnosticRows_0;
    RWStructuredBuffer<Vector<float, 4> > accumulated_0;
    RWStructuredBuffer<Vector<float, 4> > fabricRows_0;
    RWStructuredBuffer<Vector<float, 4> > wallPlane_0;
    RWStructuredBuffer<Vector<float, 4> > wallRemainder_0;
    RWStructuredBuffer<Vector<float, 4> > wallVelocity_0;
    StructuredBuffer<Vector<float, 4> > wallMaterial_0;
    StructuredBuffer<uint32_t> wallBoxIds_0;
    RWStructuredBuffer<Vector<float, 4> > wallRows_0;
    RWStructuredBuffer<Vector<float, 4> > wallAccumulated_0;
};


#line 442
struct GlobalParams_0
{
    Resources_0* r_0;
};


#line 442
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 12403 "hlsl.meta.slang"
static Vector<float, 3>  lerp_0(Vector<float, 3>  x_0, Vector<float, 3>  y_0, Vector<float, 3>  s_0)
{

#line 12415
    return x_0 + (y_0 - x_0) * s_0;
}


#line 11120
static Vector<float, 3>  floor_0(Vector<float, 3>  x_1)
{

#line 7372
    Vector<float, 3>  result_0;

#line 7372
    int32_t i_0 = int(0);

#line 7372
    for(;;)
    {

#line 7372
        if(i_0 < int(3))
        {
        }
        else
        {

#line 7372
            break;
        }

#line 7372
        result_0[i_0] = (F32_floor((_slang_vector_get_element(x_1, i_0))));

#line 7372
        i_0 = i_0 + int(1);

#line 7372
    }

#line 7372
    return result_0;
}


#line 7477
static Vector<float, 3>  abs_0(Vector<float, 3>  x_2)
{

#line 7372
    Vector<float, 3>  result_1;

#line 7372
    int32_t i_1 = int(0);

#line 7372
    for(;;)
    {

#line 7372
        if(i_1 < int(3))
        {
        }
        else
        {

#line 7372
            break;
        }

#line 7372
        result_1[i_1] = (F32_abs((_slang_vector_get_element(x_2, i_1))));

#line 7372
        i_1 = i_1 + int(1);

#line 7372
    }

#line 7372
    return result_1;
}


#line 7885
static bool any_0(Vector<bool, 3>  x_3)
{

#line 7885
    bool result_2 = false;

#line 7885
    int32_t i_2 = int(0);

#line 7945
    for(;;)
    {

#line 7945
        if(i_2 < int(3))
        {
        }
        else
        {

#line 7945
            break;
        }

#line 7946
        if(result_2)
        {

#line 7946
            result_2 = true;

#line 7946
        }
        else
        {

#line 7946
            result_2 = (bool((_slang_vector_get_element(x_3, i_2))));

#line 7946
        }

#line 7945
        i_2 = i_2 + int(1);

#line 7945
    }

    return result_2;
}


#line 10090
static float dot_0(Vector<float, 3>  x_4, Vector<float, 3>  y_1)
{

#line 10090
    int32_t i_3 = int(0);

#line 10090
    float result_3 = 0.0f;

#line 10109
    for(;;)
    {

#line 10109
        if(i_3 < int(3))
        {
        }
        else
        {

#line 10109
            break;
        }

#line 10110
        float result_4 = result_3 + _slang_vector_get_element(x_4, i_3) * _slang_vector_get_element(y_1, i_3);

#line 10109
        i_3 = i_3 + int(1);

#line 10109
        result_3 = result_4;

#line 10109
    }

    return result_3;
}


#line 12345
static float length_0(Vector<float, 3>  x_5)
{

#line 12357
    return (F32_sqrt((dot_0(x_5, x_5))));
}


#line 12898
static Vector<float, 3>  max_0(Vector<float, 3>  x_6, Vector<float, 3>  y_2)
{

#line 7378
    Vector<float, 3>  result_5;

#line 7378
    int32_t i_4 = int(0);

#line 7378
    for(;;)
    {

#line 7378
        if(i_4 < int(3))
        {
        }
        else
        {

#line 7378
            break;
        }

#line 7378
        result_5[i_4] = (F32_max((_slang_vector_get_element(x_6, i_4)), (_slang_vector_get_element(y_2, i_4))));

#line 7378
        i_4 = i_4 + int(1);

#line 7378
    }

#line 7378
    return result_5;
}


#line 13155
static Vector<float, 3>  min_0(Vector<float, 3>  x_7, Vector<float, 3>  y_3)
{

#line 7378
    Vector<float, 3>  result_6;

#line 7378
    int32_t i_5 = int(0);

#line 7378
    for(;;)
    {

#line 7378
        if(i_5 < int(3))
        {
        }
        else
        {

#line 7378
            break;
        }

#line 7378
        result_6[i_5] = (F32_min((_slang_vector_get_element(x_7, i_5)), (_slang_vector_get_element(y_3, i_5))));

#line 7378
        i_5 = i_5 + int(1);

#line 7378
    }

#line 7378
    return result_6;
}


#line 9535
static Vector<float, 3>  cross_0(Vector<float, 3>  left_0, Vector<float, 3>  right_0)
{

#line 9549
    float _S1 = left_0.y;

#line 9549
    float _S2 = right_0.z;

#line 9549
    float _S3 = left_0.z;

#line 9549
    float _S4 = right_0.y;
    float _S5 = right_0.x;

#line 9550
    float _S6 = left_0.x;

#line 9548
    return Vector<float, 3> (_S1 * _S2 - _S3 * _S4, _S3 * _S5 - _S6 * _S2, _S6 * _S4 - _S1 * _S5);
}


#line 11120
static Vector<double, 3>  floor_1(Vector<double, 3>  x_8)
{

#line 7372
    Vector<double, 3>  result_7;

#line 7372
    int32_t i_6 = int(0);

#line 7372
    for(;;)
    {

#line 7372
        if(i_6 < int(3))
        {
        }
        else
        {

#line 7372
            break;
        }

#line 7372
        result_7[i_6] = (F64_floor((_slang_vector_get_element(x_8, i_6))));

#line 7372
        i_6 = i_6 + int(1);

#line 7372
    }

#line 7372
    return result_7;
}


#line 157 "/var/folders/ff/b6vctr0n2jjfkdtcm0dt7xdr0000gp/T/sand-dem-cpu-cache/19c5b8536b5e111b21eb1c4e57be4ba2babf0ea551a25b2edbeb4ac3218400e6/solver.slang"
static Vector<float, 3>  centerBranch_0(uint32_t i_7, uint32_t j_0, KernelContext_0 * kernelContext_0)
{

#line 158
    Vector<float, 3>  _S7 = Vector<float, 3> {(*(&(kernelContext_0->globalParams_0->r_0->cell_0)[int(0)])).x, (*(&(kernelContext_0->globalParams_0->r_0->cell_0)[int(0)])).y, (*(&(kernelContext_0->globalParams_0->r_0->cell_0)[int(0)])).z};

#line 158
    Vector<double, 3>  _S8 = Vector<double, 3> {(double)_slang_vector_get_element(_S7, 0), (double)_slang_vector_get_element(_S7, 1), (double)_slang_vector_get_element(_S7, 2)};
    Vector<float, 3>  _S9 = Vector<float, 3> {(*(&(kernelContext_0->globalParams_0->r_0->position_0)[j_0])).x, (*(&(kernelContext_0->globalParams_0->r_0->position_0)[j_0])).y, (*(&(kernelContext_0->globalParams_0->r_0->position_0)[j_0])).z};

#line 159
    Vector<double, 3>  _S10 = Vector<double, 3> {(double)_slang_vector_get_element(_S9, 0), (double)_slang_vector_get_element(_S9, 1), (double)_slang_vector_get_element(_S9, 2)};

#line 159
    Vector<float, 3>  _S11 = Vector<float, 3> {(*(&(kernelContext_0->globalParams_0->r_0->position_0)[i_7])).x, (*(&(kernelContext_0->globalParams_0->r_0->position_0)[i_7])).y, (*(&(kernelContext_0->globalParams_0->r_0->position_0)[i_7])).z};

#line 159
    Vector<double, 3>  _S12 = Vector<double, 3> {(double)_slang_vector_get_element(_S11, 0), (double)_slang_vector_get_element(_S11, 1), (double)_slang_vector_get_element(_S11, 2)};
    Vector<float, 3>  _S13 = Vector<float, 3> {(*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[j_0])).x, (*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[j_0])).y, (*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[j_0])).z};

#line 160
    Vector<double, 3>  _S14 = Vector<double, 3> {(double)_slang_vector_get_element(_S13, 0), (double)_slang_vector_get_element(_S13, 1), (double)_slang_vector_get_element(_S13, 2)};

#line 160
    Vector<float, 3>  _S15 = Vector<float, 3> {(*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[i_7])).x, (*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[i_7])).y, (*(&(kernelContext_0->globalParams_0->r_0->positionRemainder_0)[i_7])).z};

#line 160
    Vector<double, 3>  _S16 = Vector<double, 3> {(double)_slang_vector_get_element(_S15, 0), (double)_slang_vector_get_element(_S15, 1), (double)_slang_vector_get_element(_S15, 2)};

#line 160
    Vector<double, 3>  d_0 = _S10 - _S12 + (_S14 - _S16);

#line 160
    Vector<double, 3>  _S17;
    if(((&kernelContext_0->globalParams_0->r_0->c_0)->periodic_0) != 0U)
    {

#line 161
        _S17 = d_0 - floor_1(d_0 / _S8 + (Vector<double, 3> )0.5) * _S8;

#line 161
    }
    else
    {

#line 161
        _S17 = d_0;

#line 161
    }

#line 161
    Vector<double, 3>  _S18 = _S17;

#line 161
    Vector<float, 3>  _S19 = Vector<float, 3> {(float)_slang_vector_get_element(_S18, 0), (float)_slang_vector_get_element(_S18, 1), (float)_slang_vector_get_element(_S18, 2)};

#line 161
    return _S19;
}


#line 112
static Vector<float, 3>  qrot_0(Vector<float, 4>  q_0, Vector<float, 3>  v_0)
{

#line 112
    Vector<float, 3>  _S20 = Vector<float, 3> {q_0.x, q_0.y, q_0.z};

#line 112
    return v_0 + (Vector<float, 3> )2.0f * cross_0(_S20, cross_0(_S20, v_0) + (Vector<float, 3> )q_0.w * v_0);
}


#line 113
static Vector<float, 3>  qinv_0(Vector<float, 4>  q_1, Vector<float, 3>  v_1)
{

#line 113
    return qrot_0(Vector<float, 4> ((- Vector<float, 3> {q_1.x, q_1.y, q_1.z}).x, (- Vector<float, 3> {q_1.x, q_1.y, q_1.z}).y, (- Vector<float, 3> {q_1.x, q_1.y, q_1.z}).z, q_1.w), v_1);
}


#line 285
static bool meshBoxMayContact_0(uint32_t shape_0, Vector<float, 3>  low_0, Vector<float, 3>  high_0, KernelContext_0 * kernelContext_1)
{

#line 286
    uint32_t root_0 = kernelContext_1->globalParams_0->r_0->meshRoots_0.Load(shape_0).x;
    Vector<float, 3>  targetLow_0 = Vector<float, 3> {kernelContext_1->globalParams_0->r_0->meshMin_0.Load(root_0).x, kernelContext_1->globalParams_0->r_0->meshMin_0.Load(root_0).y, kernelContext_1->globalParams_0->r_0->meshMin_0.Load(root_0).z};

#line 287
    Vector<float, 3>  targetHigh_0 = Vector<float, 3> {kernelContext_1->globalParams_0->r_0->meshMax_0.Load(root_0).x, kernelContext_1->globalParams_0->r_0->meshMax_0.Load(root_0).y, kernelContext_1->globalParams_0->r_0->meshMax_0.Load(root_0).z};

#line 287
    bool _S21;
    if(any_0(high_0 < targetLow_0))
    {

#line 288
        _S21 = true;

#line 288
    }
    else
    {

#line 288
        _S21 = any_0(low_0 > targetHigh_0);

#line 288
    }

#line 288
    if(_S21)
    {

#line 288
        return false;
    }

#line 289
    Vector<float, 3>  center_0 = (Vector<float, 3> )0.5f * (low_0 + high_0);

#line 289
    Vector<float, 3>  _S22 = (Vector<float, 3> )0.5f * (high_0 - low_0);
    Vector<float, 3>  direction_0 = (center_0 - (Vector<float, 3> )0.5f * (targetLow_0 + targetHigh_0)) / (targetHigh_0 - targetLow_0);
    Vector<float, 3>  a_0 = abs_0(direction_0);

#line 291
    float _S23 = a_0.x;

#line 291
    float _S24 = a_0.y;

#line 291
    if(_S23 >= _S24)
    {

#line 291
        _S21 = _S23 >= (a_0.z);

#line 291
    }
    else
    {

#line 291
        _S21 = false;

#line 291
    }

#line 291
    int32_t _S25;

#line 291
    if(_S21)
    {

#line 291
        _S25 = int(0);

#line 291
    }
    else
    {

#line 291
        if(_S24 >= (a_0.z))
        {

#line 291
            _S25 = int(1);

#line 291
        }
        else
        {

#line 291
            _S25 = int(2);

#line 291
        }

#line 291
    }

#line 291
    uint32_t axis_0 = uint32_t(_S25);
    uint32_t _S26 = 2U * axis_0;

#line 292
    if(_slang_vector_get_element(direction_0, axis_0) >= 0.0f)
    {

#line 292
        _S25 = int(1);

#line 292
    }
    else
    {

#line 292
        _S25 = int(0);

#line 292
    }
    Vector<uint32_t, 4>  _S27 = kernelContext_1->globalParams_0->r_0->meshHullInfo_0.Load(6U * shape_0 + (_S26 + uint32_t(_S25)));

#line 293
    uint32_t i_8 = 0U;
    for(;;)
    {

#line 294
        if(i_8 < (_S27.y))
        {
        }
        else
        {

#line 294
            break;
        }

#line 295
        Vector<float, 4>  plane_0 = kernelContext_1->globalParams_0->r_0->meshPlanes_0.Load(_S27.x + i_8);
        Vector<float, 3>  _S28 = Vector<float, 3> {plane_0.x, plane_0.y, plane_0.z};

#line 296
        if((dot_0(_S28, center_0) + plane_0.w - dot_0(abs_0(_S28), _S22)) > 0.0f)
        {

#line 296
            return false;
        }

#line 294
        i_8 = i_8 + 1U;

#line 294
    }



    return true;
}

static bool meshClusterMayContact_0(uint32_t src_0, uint32_t dst_0, uint32_t cluster_0, KernelContext_0 * kernelContext_2)
{
++profile_calls[3];

#line 302
    Vector<float, 3>  _S29 = Vector<float, 3> {kernelContext_2->globalParams_0->r_0->meshClusterMin_0.Load(cluster_0).x, kernelContext_2->globalParams_0->r_0->meshClusterMin_0.Load(cluster_0).y, kernelContext_2->globalParams_0->r_0->meshClusterMin_0.Load(cluster_0).z};

#line 302
    Vector<float, 3>  _S30 = Vector<float, 3> {kernelContext_2->globalParams_0->r_0->meshClusterMax_0.Load(cluster_0).x, kernelContext_2->globalParams_0->r_0->meshClusterMax_0.Load(cluster_0).y, kernelContext_2->globalParams_0->r_0->meshClusterMax_0.Load(cluster_0).z};

#line 302
    Vector<float, 3>  _S31 = centerBranch_0(src_0, dst_0, kernelContext_2);
    Vector<float, 3>  _S32 = (Vector<float, 3> )1.00000001504746622e+30f;

#line 303
    Vector<float, 3>  _S33 = (Vector<float, 3> )-1.00000001504746622e+30f;

#line 303
    uint32_t corner_0 = 0U;

#line 303
    Vector<float, 3>  low_1 = _S32;

#line 303
    Vector<float, 3>  high_1 = _S33;
    for(;;)
    {

#line 304
        if(corner_0 < 8U)
        {
        }
        else
        {

#line 304
            break;
        }

#line 304
        float _S34;
        if((corner_0 & 1U) != 0U)
        {

#line 305
            _S34 = _S30.x;

#line 305
        }
        else
        {

#line 305
            _S34 = _S29.x;

#line 305
        }

#line 305
        float _S35;
        if((corner_0 & 2U) != 0U)
        {

#line 306
            _S35 = _S30.y;

#line 306
        }
        else
        {

#line 306
            _S35 = _S29.y;

#line 306
        }

#line 306
        float _S36;
        if((corner_0 & 4U) != 0U)
        {

#line 307
            _S36 = _S30.z;

#line 307
        }
        else
        {

#line 307
            _S36 = _S29.z;

#line 307
        }
        Vector<float, 3>  p_0 = qinv_0(*(&(kernelContext_2->globalParams_0->r_0->rotation_0)[dst_0]), qrot_0(*(&(kernelContext_2->globalParams_0->r_0->rotation_0)[src_0]), Vector<float, 3> (_S34, _S35, _S36) * (Vector<float, 3> )kernelContext_2->globalParams_0->r_0->body_0.Load(src_0).x) - _S31) / (Vector<float, 3> )kernelContext_2->globalParams_0->r_0->body_0.Load(dst_0).x;
        Vector<float, 3>  _S37 = min_0(low_1, p_0);

#line 309
        Vector<float, 3>  _S38 = max_0(high_1, p_0);

#line 304
        corner_0 = corner_0 + 1U;

#line 304
        low_1 = _S37;

#line 304
        high_1 = _S38;

#line 304
    }

#line 312
    float pad_0 = 0.00000762939453125f * (1.0f + (F32_max(((F32_max(((F32_abs((low_1.x)))), ((F32_abs((low_1.y))))))), ((F32_abs((low_1.z)))))) + (F32_max(((F32_max(((F32_abs((high_1.x)))), ((F32_abs((high_1.y))))))), ((F32_abs((high_1.z)))))) + length_0(_S31) / kernelContext_2->globalParams_0->r_0->body_0.Load(dst_0).x);

#line 312
    bool _S39 = meshBoxMayContact_0(kernelContext_2->globalParams_0->r_0->shapeId_0.Load(dst_0), low_1 - (Vector<float, 3> )pad_0, high_1 + (Vector<float, 3> )pad_0, kernelContext_2);


    return _S39;
}


#line 270
static bool meshMayContain_0(uint32_t shape_1, Vector<float, 3>  p_1, KernelContext_0 * kernelContext_3)
{
++profile_calls[2];

#line 271
    uint32_t root_1 = kernelContext_3->globalParams_0->r_0->meshRoots_0.Load(shape_1).x;
    Vector<float, 3>  low_2 = Vector<float, 3> {kernelContext_3->globalParams_0->r_0->meshMin_0.Load(root_1).x, kernelContext_3->globalParams_0->r_0->meshMin_0.Load(root_1).y, kernelContext_3->globalParams_0->r_0->meshMin_0.Load(root_1).z};

#line 272
    Vector<float, 3>  high_2 = Vector<float, 3> {kernelContext_3->globalParams_0->r_0->meshMax_0.Load(root_1).x, kernelContext_3->globalParams_0->r_0->meshMax_0.Load(root_1).y, kernelContext_3->globalParams_0->r_0->meshMax_0.Load(root_1).z};

#line 272
    bool _S40;
    if(any_0(p_1 < low_2))
    {

#line 273
        _S40 = true;

#line 273
    }
    else
    {

#line 273
        _S40 = any_0(p_1 > high_2);

#line 273
    }

#line 273
    if(_S40)
    {

#line 273
        return false;
    }

#line 274
    Vector<float, 3>  direction_1 = (p_1 - (Vector<float, 3> )0.5f * (low_2 + high_2)) / (high_2 - low_2);
    Vector<float, 3>  a_1 = abs_0(direction_1);

#line 275
    float _S41 = a_1.x;

#line 275
    float _S42 = a_1.y;

#line 275
    if(_S41 >= _S42)
    {

#line 275
        _S40 = _S41 >= (a_1.z);

#line 275
    }
    else
    {

#line 275
        _S40 = false;

#line 275
    }

#line 275
    int32_t _S43;

#line 275
    if(_S40)
    {

#line 275
        _S43 = int(0);

#line 275
    }
    else
    {

#line 275
        if(_S42 >= (a_1.z))
        {

#line 275
            _S43 = int(1);

#line 275
        }
        else
        {

#line 275
            _S43 = int(2);

#line 275
        }

#line 275
    }

#line 275
    uint32_t axis_1 = uint32_t(_S43);
    uint32_t _S44 = 2U * axis_1;

#line 276
    if(_slang_vector_get_element(direction_1, axis_1) >= 0.0f)
    {

#line 276
        _S43 = int(1);

#line 276
    }
    else
    {

#line 276
        _S43 = int(0);

#line 276
    }
    Vector<uint32_t, 4>  _S45 = kernelContext_3->globalParams_0->r_0->meshHullInfo_0.Load(6U * shape_1 + (_S44 + uint32_t(_S43)));

#line 277
    uint32_t i_9 = 0U;
    for(;;)
    {

#line 278
        if(i_9 < (_S45.y))
        {
        }
        else
        {

#line 278
            break;
        }

#line 279
        Vector<float, 4>  plane_1 = kernelContext_3->globalParams_0->r_0->meshPlanes_0.Load(_S45.x + i_9);
        if((dot_0(Vector<float, 3> {plane_1.x, plane_1.y, plane_1.z}, p_1) + plane_1.w) > 0.0f)
        {

#line 280
            return false;
        }

#line 278
        i_9 = i_9 + 1U;

#line 278
    }



    return true;
}


#line 229
static float meshBoxDistance_0(uint32_t node_0, Vector<float, 3>  p_2, KernelContext_0 * kernelContext_4)
{
++profile_calls[4];

#line 230
    Vector<int32_t, 3>  _S46 = (Vector<int32_t, 3> )int(0);

#line 230
    Vector<float, 3>  _S47 = Vector<float, 3> {(float)_slang_vector_get_element(_S46, 0), (float)_slang_vector_get_element(_S46, 1), (float)_slang_vector_get_element(_S46, 2)};

#line 230
    Vector<float, 3>  d_1 = max_0(max_0(Vector<float, 3> {kernelContext_4->globalParams_0->r_0->meshMin_0.Load(node_0).x, kernelContext_4->globalParams_0->r_0->meshMin_0.Load(node_0).y, kernelContext_4->globalParams_0->r_0->meshMin_0.Load(node_0).z} - p_2, p_2 - Vector<float, 3> {kernelContext_4->globalParams_0->r_0->meshMax_0.Load(node_0).x, kernelContext_4->globalParams_0->r_0->meshMax_0.Load(node_0).y, kernelContext_4->globalParams_0->r_0->meshMax_0.Load(node_0).z}), _S47);
    return dot_0(d_1, d_1);
}


#line 231
static void cpuAtomicOr_0(int32_t _S48, uint32_t _S49, uint32_t * _S50, KernelContext_0 * kernelContext_5)
{

#line 231
    uint32_t * _S51 = (&(kernelContext_5->globalParams_0->r_0->control_0)[_S48]);

#line 4
    *_S50 = *_S51;

#line 4
    *_S51 = (*_S51) | _S49;

#line 4
    return;
}


#line 192
static void fail_0(uint32_t bit_0, KernelContext_0 * kernelContext_6)
{

#line 192
    uint32_t unused_0;

#line 192
    cpuAtomicOr_0(int(0), bit_0, &unused_0, kernelContext_6);

#line 192
    return;
}


#line 212
static Vector<float, 3>  closestTriangle_0(Vector<float, 3>  p_3, Vector<float, 3>  a_2, Vector<float, 3>  b_0, Vector<float, 3>  c_1, uint32_t * feature_0)
{
++profile_calls[5];

#line 213
    Vector<float, 3>  ab_0 = b_0 - a_2;

#line 213
    Vector<float, 3>  ac_0 = c_1 - a_2;

#line 213
    Vector<float, 3>  ap_0 = p_3 - a_2;
    float d1_0 = dot_0(ab_0, ap_0);

#line 214
    float d2_0 = dot_0(ac_0, ap_0);

#line 214
    *feature_0 = 4U;

#line 214
    bool _S52;
    if(d1_0 <= 0.0f)
    {

#line 215
        _S52 = d2_0 <= 0.0f;

#line 215
    }
    else
    {

#line 215
        _S52 = false;

#line 215
    }

#line 215
    if(_S52)
    {

#line 215
        return a_2;
    }

#line 216
    Vector<float, 3>  bp_0 = p_3 - b_0;

#line 216
    float d3_0 = dot_0(ab_0, bp_0);

#line 216
    float d4_0 = dot_0(ac_0, bp_0);
    if(d3_0 >= 0.0f)
    {

#line 217
        _S52 = d4_0 <= d3_0;

#line 217
    }
    else
    {

#line 217
        _S52 = false;

#line 217
    }

#line 217
    if(_S52)
    {

#line 217
        *feature_0 = 5U;

#line 217
        return b_0;
    }

#line 218
    float vc_0 = d1_0 * d4_0 - d3_0 * d2_0;
    if(vc_0 <= 0.0f)
    {

#line 219
        _S52 = d1_0 >= 0.0f;

#line 219
    }
    else
    {

#line 219
        _S52 = false;

#line 219
    }

#line 219
    if(_S52)
    {

#line 219
        _S52 = d3_0 <= 0.0f;

#line 219
    }
    else
    {

#line 219
        _S52 = false;

#line 219
    }

#line 219
    if(_S52)
    {

#line 219
        *feature_0 = 1U;

#line 219
        return a_2 + (Vector<float, 3> )(d1_0 / (d1_0 - d3_0)) * ab_0;
    }

#line 220
    Vector<float, 3>  cp_0 = p_3 - c_1;

#line 220
    float d5_0 = dot_0(ab_0, cp_0);

#line 220
    float d6_0 = dot_0(ac_0, cp_0);
    if(d6_0 >= 0.0f)
    {

#line 221
        _S52 = d5_0 <= d6_0;

#line 221
    }
    else
    {

#line 221
        _S52 = false;

#line 221
    }

#line 221
    if(_S52)
    {

#line 221
        *feature_0 = 6U;

#line 221
        return c_1;
    }

#line 222
    float vb_0 = d5_0 * d2_0 - d1_0 * d6_0;
    if(vb_0 <= 0.0f)
    {

#line 223
        _S52 = d2_0 >= 0.0f;

#line 223
    }
    else
    {

#line 223
        _S52 = false;

#line 223
    }

#line 223
    if(_S52)
    {

#line 223
        _S52 = d6_0 <= 0.0f;

#line 223
    }
    else
    {

#line 223
        _S52 = false;

#line 223
    }

#line 223
    if(_S52)
    {

#line 223
        *feature_0 = 3U;

#line 223
        return a_2 + (Vector<float, 3> )(d2_0 / (d2_0 - d6_0)) * ac_0;
    }

#line 224
    float va_0 = d3_0 * d6_0 - d5_0 * d4_0;
    if(va_0 <= 0.0f)
    {

#line 225
        _S52 = (d4_0 - d3_0) >= 0.0f;

#line 225
    }
    else
    {

#line 225
        _S52 = false;

#line 225
    }

#line 225
    if(_S52)
    {

#line 225
        _S52 = (d5_0 - d6_0) >= 0.0f;

#line 225
    }
    else
    {

#line 225
        _S52 = false;

#line 225
    }

#line 225
    if(_S52)
    {

#line 225
        *feature_0 = 2U;

#line 225
        float _S53 = d4_0 - d3_0;

#line 225
        return b_0 + (Vector<float, 3> )(_S53 / (_S53 + (d5_0 - d6_0))) * (c_1 - b_0);
    }

#line 226
    float inv_0 = 1.0f / (va_0 + vb_0 + vc_0);

#line 226
    *feature_0 = 0U;
    return a_2 + ab_0 * (Vector<float, 3> )(vb_0 * inv_0) + ac_0 * (Vector<float, 3> )(vc_0 * inv_0);
}




static Vector<float, 4>  meshDistance_0(uint32_t shape_2, Vector<float, 3>  p_4, bool surfaceNormal_0, KernelContext_0 * kernelContext_7)
{
++profile_calls[1];

#line 233
    float da_0;

#line 233
    Vector<float, 3>  nearest_0;
    uint32_t root_2 = kernelContext_7->globalParams_0->r_0->meshRoots_0.Load(shape_2).x;
    FixedArray<uint32_t, 32>  pending_0;

#line 235
    pending_0[int(0)] = root_2;

#line 235
    FixedArray<float, 32>  pendingDistance_0;

#line 235
    float _S54 = meshBoxDistance_0(root_2, p_4, kernelContext_7);
    pendingDistance_0[int(0)] = _S54;
    Vector<int32_t, 3>  _S55 = (Vector<int32_t, 3> )int(0);

#line 237
    Vector<float, 3>  _S56 = Vector<float, 3> {(float)_slang_vector_get_element(_S55, 0), (float)_slang_vector_get_element(_S55, 1), (float)_slang_vector_get_element(_S55, 2)};

#line 237
    float best_0 = 1.00000001504746622e+30f;

#line 237
    bool found_0 = false;

#line 237
    uint32_t nearestNormal_0 = 0U;

#line 237
    Vector<float, 3>  nearest_1 = _S56;

#line 237
    uint32_t top_0 = 1U;
    for(;;)
    {

#line 238
        if(top_0 > 0U)
        {
        }
        else
        {

#line 238
            break;
        }

#line 239
        uint32_t top_1 = top_0 - 1U;

#line 239
        uint32_t node_1 = pending_0[top_1];
        if((pendingDistance_0[top_1]) > best_0)
        {

#line 240
            top_0 = top_1;

#line 240
            continue;
        }

#line 241
        Vector<uint32_t, 4>  link_0 = kernelContext_7->globalParams_0->r_0->meshLinks_0.Load(node_1);
        uint32_t _S57 = link_0.z;

#line 242
        uint32_t a_3;

#line 242
        uint32_t b_1;

#line 242
        if(_S57 == 0U)
        {

#line 243
            uint32_t a_4 = node_1 + 1U;

#line 243
            uint32_t b_2 = kernelContext_7->globalParams_0->r_0->meshLinks_0.Load(a_4).x;

#line 243
            float _S58 = meshBoxDistance_0(a_4, p_4, kernelContext_7);

#line 243
            float _S59 = meshBoxDistance_0(b_2, p_4, kernelContext_7);

#line 243
            float db_0;

            if(_S58 < _S59)
            {

#line 245
                da_0 = _S59;

#line 245
                a_3 = b_2;

#line 245
                db_0 = _S58;

#line 245
                b_1 = a_4;

#line 245
            }
            else
            {

#line 245
                da_0 = _S58;

#line 245
                a_3 = a_4;

#line 245
                db_0 = _S59;

#line 245
                b_1 = b_2;

#line 245
            }

            if((top_1 + 2U) > 32U)
            {

#line 247
                fail_0(1024U, kernelContext_7);

#line 247
                return Vector<float, 4> (0.0f, 0.0f, 0.0f, 1.00000002004087734e+20f);
            }

#line 247
            uint32_t top_2;
            if(da_0 <= best_0)
            {

#line 248
                pending_0[top_1] = a_3;

#line 248
                uint32_t _S60 = top_1 + 1U;

#line 248
                pendingDistance_0[top_1] = da_0;

#line 248
                top_2 = _S60;

#line 248
            }
            else
            {

#line 248
                top_2 = top_1;

#line 248
            }
            if(db_0 <= best_0)
            {

#line 249
                pending_0[top_2] = b_1;

#line 249
                uint32_t _S61 = top_2 + 1U;

#line 249
                pendingDistance_0[top_2] = db_0;

#line 249
                top_0 = _S61;

#line 249
            }
            else
            {

#line 249
                top_0 = top_2;

#line 249
            }
            continue;
        }

#line 250
        da_0 = best_0;

#line 250
        bool found_1 = found_0;

#line 250
        a_3 = nearestNormal_0;

#line 250
        nearest_0 = nearest_1;

#line 250
        b_1 = 0U;

        for(;;)
        {

#line 252
            if(b_1 < _S57)
            {
            }
            else
            {

#line 252
                break;
            }

#line 253
            uint32_t triangle_0 = link_0.y + b_1;
            uint32_t _S62 = 3U * triangle_0;

#line 253
            uint32_t feature_1;
            Vector<float, 3>  q_2 = closestTriangle_0(p_4, Vector<float, 3> {kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62).x, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62).y, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62).z}, Vector<float, 3> {kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 1U).x, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 1U).y, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 1U).z}, Vector<float, 3> {kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 2U).x, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 2U).y, kernelContext_7->globalParams_0->r_0->meshTriangles_0.Load(_S62 + 2U).z}, &feature_1);
            Vector<float, 3>  _S63 = p_4 - q_2;

#line 255
            float distance_1 = dot_0(_S63, _S63);
            if(distance_1 < da_0)
            {

#line 256
                uint32_t _S64 = 7U * triangle_0 + feature_1;

#line 256
                da_0 = distance_1;

#line 256
                found_1 = true;

#line 256
                a_3 = _S64;

#line 256
                nearest_0 = q_2;

#line 256
            }

#line 252
            b_1 = b_1 + 1U;

#line 252
        }

#line 252
        best_0 = da_0;

#line 252
        found_0 = found_1;

#line 252
        nearestNormal_0 = a_3;

#line 252
        nearest_1 = nearest_0;

#line 252
        top_0 = top_1;

#line 238
    }

#line 259
    if(found_0)
    {

#line 259
        nearest_0 = Vector<float, 3> {kernelContext_7->globalParams_0->r_0->meshNormals_0.Load(nearestNormal_0).x, kernelContext_7->globalParams_0->r_0->meshNormals_0.Load(nearestNormal_0).y, kernelContext_7->globalParams_0->r_0->meshNormals_0.Load(nearestNormal_0).z};

#line 259
    }
    else
    {

#line 259
        nearest_0 = _S56;

#line 259
    }
    float distance_2 = (F32_sqrt((best_0)));

#line 260
    Vector<float, 3>  _S65 = p_4 - nearest_1;

#line 260
    if((dot_0(_S65, nearest_0)) < 0.0f)
    {

#line 260
        da_0 = -1.0f;

#line 260
    }
    else
    {

#line 260
        da_0 = 1.0f;

#line 260
    }

#line 260
    Vector<float, 3>  normal_0;
    if(surfaceNormal_0)
    {

#line 261
        normal_0 = nearest_0;

#line 261
    }
    else
    {

#line 261
        if(distance_2 > 9.999999960041972e-13f)
        {

#line 261
            nearest_1 = (Vector<float, 3> )da_0 * _S65 / (Vector<float, 3> )distance_2;

#line 261
        }
        else
        {

#line 261
            nearest_1 = nearest_0;

#line 261
        }

#line 261
        normal_0 = nearest_1;

#line 261
    }
    return Vector<float, 4> (normal_0.x, normal_0.y, normal_0.z, da_0 * distance_2);
}


#line 197
static Vector<float, 4>  sdf_0(uint32_t s_1, Vector<float, 3>  p_5, KernelContext_0 * kernelContext_8)
{

#line 198
    Vector<uint32_t, 4>  meta_1 = kernelContext_8->globalParams_0->r_0->grids_0.Load(s_1);

#line 198
    Vector<float, 4>  o_0 = kernelContext_8->globalParams_0->r_0->gridOrigin_0.Load(s_1);
    float _S66 = o_0.w;

#line 199
    Vector<float, 3>  g_0 = (p_5 - Vector<float, 3> {o_0.x, o_0.y, o_0.z}) / (Vector<float, 3> )_S66;

#line 199
    Vector<float, 3>  _S67 = floor_0(g_0);

#line 199
    Vector<int32_t, 3>  _S68 = Vector<int32_t, 3> {(int32_t)_slang_vector_get_element(_S67, 0), (int32_t)_slang_vector_get_element(_S67, 1), (int32_t)_slang_vector_get_element(_S67, 2)};

#line 199
    bool _S69;
    if(any_0(_S68 < ((Vector<int32_t, 3> )int(0))))
    {

#line 200
        _S69 = true;

#line 200
    }
    else
    {

#line 200
        Vector<uint32_t, 3>  _S70 = Vector<uint32_t, 3> {meta_1.y, meta_1.z, meta_1.w};

#line 200
        Vector<int32_t, 3>  _S71 = Vector<int32_t, 3> {(int32_t)_slang_vector_get_element(_S70, 0), (int32_t)_slang_vector_get_element(_S70, 1), (int32_t)_slang_vector_get_element(_S70, 2)};

#line 200
        _S69 = any_0(_S68 >= (_S71 - (Vector<int32_t, 3> )int(1)));

#line 200
    }

#line 200
    if(_S69)
    {

#line 200
        return Vector<float, 4> (0.0f, 0.0f, 0.0f, 1.00000002004087734e+20f);
    }

#line 201
    Vector<float, 3>  _S72 = Vector<float, 3> {(float)_slang_vector_get_element(_S68, 0), (float)_slang_vector_get_element(_S68, 1), (float)_slang_vector_get_element(_S68, 2)};

#line 201
    Vector<float, 3>  _S73 = g_0 - _S72;

#line 201
    Vector<int32_t, 3>  _S74 = (Vector<int32_t, 3> )int(0);

#line 201
    Vector<float, 3>  _S75 = Vector<float, 3> {(float)_slang_vector_get_element(_S74, 0), (float)_slang_vector_get_element(_S74, 1), (float)_slang_vector_get_element(_S74, 2)};

#line 201
    uint32_t i_10 = 0U;

#line 201
    float value_0 = 0.0f;

#line 201
    Vector<float, 3>  grad_0 = _S75;
    for(;;)
    {

#line 202
        if(i_10 < 2U)
        {
        }
        else
        {

#line 202
            break;
        }

#line 202
        uint32_t j_1 = 0U;

#line 202
        for(;;)
        {

#line 202
            if(j_1 < 2U)
            {
            }
            else
            {

#line 202
                break;
            }

#line 202
            uint32_t k_0 = 0U;

#line 202
            float value_1 = value_0;

#line 202
            Vector<float, 3>  grad_1 = grad_0;

#line 202
            for(;;)
            {

#line 202
                if(k_0 < 2U)
                {
                }
                else
                {

#line 202
                    break;
                }

#line 203
                Vector<uint32_t, 3>  b_3 = Vector<uint32_t, 3> (i_10, j_1, k_0);

#line 203
                Vector<float, 3>  _S76 = Vector<float, 3> {(float)_slang_vector_get_element(b_3, 0), (float)_slang_vector_get_element(b_3, 1), (float)_slang_vector_get_element(b_3, 2)};

#line 203
                Vector<float, 3>  w_0 = lerp_0((Vector<float, 3> )1.0f - _S73, _S73, _S76);

#line 203
                Vector<uint32_t, 3>  _S77 = Vector<uint32_t, 3> {(uint32_t)_slang_vector_get_element(_S68, 0), (uint32_t)_slang_vector_get_element(_S68, 1), (uint32_t)_slang_vector_get_element(_S68, 2)};

#line 203
                Vector<uint32_t, 3>  index_0 = _S77 + b_3;
                float f_0 = kernelContext_8->globalParams_0->r_0->distance_0.Load(meta_1.x + (index_0.x * meta_1.z + index_0.y) * meta_1.w + index_0.z);
                float _S78 = w_0.x;

#line 205
                float _S79 = w_0.y;

#line 205
                float _S80 = w_0.z;

#line 205
                float value_2 = value_1 + f_0 * _S78 * _S79 * _S80;
                Vector<float, 3>  grad_2 = grad_1 + (Vector<float, 3> )f_0 * Vector<float, 3> ((2.0f * float(i_10) - 1.0f) * _S79 * _S80, (2.0f * float(j_1) - 1.0f) * _S78 * _S80, (2.0f * float(k_0) - 1.0f) * _S78 * _S79) / (Vector<float, 3> )_S66;

#line 202
                k_0 = k_0 + 1U;

#line 202
                value_1 = value_2;

#line 202
                grad_1 = grad_2;

#line 202
            }

#line 202
            j_1 = j_1 + 1U;

#line 202
            value_0 = value_1;

#line 202
            grad_0 = grad_1;

#line 202
        }

#line 202
        i_10 = i_10 + 1U;

#line 202
    }

#line 208
    return Vector<float, 4> (grad_0.x, grad_0.y, grad_0.z, value_0);
}


#line 330
static bool nodeContact_0(uint32_t src_1, uint32_t dst_1, uint32_t node_2, Vector<float, 4>  * pd_0, Vector<float, 4>  * na_0, Vector<float, 4>  rejectionPlane_0, KernelContext_0 * kernelContext_9)
{
++profile_calls[0];

#line 331
    uint32_t sid_0 = kernelContext_9->globalParams_0->r_0->shapeId_0.Load(src_1);

#line 331
    Vector<float, 4>  sp_0 = kernelContext_9->globalParams_0->r_0->sample_0.Load(kernelContext_9->globalParams_0->r_0->sampleInfo_0.Load(sid_0).x + node_2);
    Vector<float, 3>  _S81 = Vector<float, 3> {sp_0.x, sp_0.y, sp_0.z};

#line 332
    Vector<float, 3>  arm_0 = qrot_0(*(&(kernelContext_9->globalParams_0->r_0->rotation_0)[src_1]), _S81 * (Vector<float, 3> )kernelContext_9->globalParams_0->r_0->body_0.Load(src_1).x);

#line 332
    Vector<float, 3>  _S82 = centerBranch_0(src_1, dst_1, kernelContext_9);

    Vector<float, 3>  local_0 = qinv_0(*(&(kernelContext_9->globalParams_0->r_0->rotation_0)[dst_1]), arm_0 - _S82) / (Vector<float, 3> )kernelContext_9->globalParams_0->r_0->body_0.Load(dst_1).x;
    uint32_t targetShape_0 = kernelContext_9->globalParams_0->r_0->shapeId_0.Load(dst_1);

#line 335
    Vector<float, 4>  f_1;
    if(((&kernelContext_9->globalParams_0->r_0->c_0)->meshGeometry_0) != 0U)
    {
        if((dot_0(Vector<float, 3> {rejectionPlane_0.x, rejectionPlane_0.y, rejectionPlane_0.z}, local_0) + rejectionPlane_0.w) > (0.00000762939453125f * (1.0f + (F32_max(((F32_max(((F32_abs((local_0.x)))), ((F32_abs((local_0.y))))))), ((F32_abs((local_0.z)))))))))
        {

#line 338
            Vector<int32_t, 4>  _S83 = (Vector<int32_t, 4> )int(0);

#line 338
            Vector<float, 4>  _S84 = Vector<float, 4> {(float)_slang_vector_get_element(_S83, 0), (float)_slang_vector_get_element(_S83, 1), (float)_slang_vector_get_element(_S83, 2), (float)_slang_vector_get_element(_S83, 3)};

#line 338
            *pd_0 = _S84;

#line 338
            *na_0 = _S84;

#line 338
            return false;
        }

#line 338
        bool _S85 = meshMayContain_0(targetShape_0, local_0, kernelContext_9);
        if(!_S85)
        {

#line 339
            Vector<int32_t, 4>  _S86 = (Vector<int32_t, 4> )int(0);

#line 339
            Vector<float, 4>  _S87 = Vector<float, 4> {(float)_slang_vector_get_element(_S86, 0), (float)_slang_vector_get_element(_S86, 1), (float)_slang_vector_get_element(_S86, 2), (float)_slang_vector_get_element(_S86, 3)};

#line 339
            *pd_0 = _S87;

#line 339
            *na_0 = _S87;

#line 339
            return false;
        }

#line 339
        Vector<float, 4>  _S88 = meshDistance_0(targetShape_0, local_0, false, kernelContext_9);

#line 339
        f_1 = _S88;

#line 336
    }
    else
    {

#line 336
        Vector<float, 4>  _S89 = sdf_0(targetShape_0, local_0, kernelContext_9);

#line 336
        f_1 = _S89;

#line 336
    }

#line 342
    *pd_0 = Vector<float, 4> (arm_0.x, arm_0.y, arm_0.z, - f_1.w * kernelContext_9->globalParams_0->r_0->body_0.Load(dst_1).x);

#line 342
    Vector<int32_t, 4>  _S90 = (Vector<int32_t, 4> )int(0);

#line 342
    Vector<float, 4>  _S91 = Vector<float, 4> {(float)_slang_vector_get_element(_S90, 0), (float)_slang_vector_get_element(_S90, 1), (float)_slang_vector_get_element(_S90, 2), (float)_slang_vector_get_element(_S90, 3)};

#line 342
    *na_0 = _S91;
    if(((*pd_0).w) <= 0.0f)
    {

#line 343
        return false;
    }

#line 344
    Vector<float, 3>  grad_3 = qrot_0(*(&(kernelContext_9->globalParams_0->r_0->rotation_0)[dst_1]), Vector<float, 3> {f_1.x, f_1.y, f_1.z});

#line 344
    float _S92 = length_0(grad_3);

#line 344
    float len_0;

#line 344
    Vector<float, 3>  grad_4;
    if(((&kernelContext_9->globalParams_0->r_0->c_0)->sourceSurfaceNormal_0) != 0U)
    {

#line 345
        Vector<float, 4>  _S93 = meshDistance_0(sid_0, _S81, true, kernelContext_9);



        if((F32_abs((_S93.w))) > 0.00000999999974738f)
        {

#line 349
            fail_0(2U, kernelContext_9);

#line 349
            return false;
        }

#line 350
        Vector<float, 3>  grad_5 = - qrot_0(*(&(kernelContext_9->globalParams_0->r_0->rotation_0)[src_1]), Vector<float, 3> {_S93.x, _S93.y, _S93.z});

#line 350
        len_0 = length_0(grad_5);

#line 350
        grad_4 = grad_5;

#line 345
    }
    else
    {

#line 345
        len_0 = _S92;

#line 345
        grad_4 = grad_3;

#line 345
    }

#line 345
    bool _S94;

#line 352
    if(len_0 < 1.00000001168609742e-07f)
    {

#line 352
        _S94 = true;

#line 352
    }
    else
    {

#line 352
        _S94 = !(F32_isfinite(((*pd_0).w)));

#line 352
    }

#line 352
    if(_S94)
    {

#line 352
        fail_0(2U, kernelContext_9);

#line 352
        return false;
    }

#line 353
    *na_0 = Vector<float, 4> ((grad_4 / (Vector<float, 3> )len_0).x, (grad_4 / (Vector<float, 3> )len_0).y, (grad_4 / (Vector<float, 3> )len_0).z, sp_0.w * kernelContext_9->globalParams_0->r_0->body_0.Load(src_1).x * kernelContext_9->globalParams_0->r_0->body_0.Load(src_1).x);
    if(((*pd_0).w) > ((&kernelContext_9->globalParams_0->r_0->c_0)->maxOverlap_0 * (F32_min((kernelContext_9->globalParams_0->r_0->body_0.Load(src_1).w), (kernelContext_9->globalParams_0->r_0->body_0.Load(dst_1).w)))))
    {

#line 354
        fail_0(4U, kernelContext_9);

#line 354
    }
    return true;
}


#line 441
static void cpuCollectMesh_0(uint32_t pair_0, KernelContext_0 * kernelContext_10)
{

#line 441
    uint32_t plane_2;

#line 441
    bool _S95;
    if(pair_0 >= (U32_min((*(&(kernelContext_10->globalParams_0->r_0->control_0)[int(1)])), ((&kernelContext_10->globalParams_0->r_0->c_0)->pairCapacity_0))))
    {

#line 442
        _S95 = true;

#line 442
    }
    else
    {

#line 442
        _S95 = (*(&(kernelContext_10->globalParams_0->r_0->control_0)[int(0)])) != 0U;

#line 442
    }

#line 442
    if(_S95)
    {

#line 442
        return;
    }

#line 443
    Vector<uint32_t, 2>  _S96 = *(&(kernelContext_10->globalParams_0->r_0->pairs_0)[pair_0]);

#line 443
    uint32_t pass_0 = 0U;
    for(;;)
    {

#line 444
        if(pass_0 < 2U)
        {
        }
        else
        {

#line 444
            break;
        }

#line 445
        bool _S97 = pass_0 == 0U;

#line 445
        uint32_t i_11;

#line 445
        if(_S97)
        {

#line 445
            i_11 = _S96.x;

#line 445
        }
        else
        {

#line 445
            i_11 = _S96.y;

#line 445
        }

#line 445
        uint32_t j_2;

#line 445
        if(_S97)
        {

#line 445
            j_2 = _S96.y;

#line 445
        }
        else
        {

#line 445
            j_2 = _S96.x;

#line 445
        }

#line 450
        Vector<float, 4>  _S98 = *(&(kernelContext_10->globalParams_0->r_0->rotation_0)[j_2]);

#line 450
        Vector<float, 3>  _S99 = centerBranch_0(i_11, j_2, kernelContext_10);

#line 450
        Vector<float, 3>  _S100 = - qinv_0(_S98, _S99) / (Vector<float, 3> )kernelContext_10->globalParams_0->r_0->body_0.Load(j_2).x;
        Vector<int32_t, 4>  _S101 = (Vector<int32_t, 4> )int(0);

#line 451
        Vector<float, 4>  _S102 = Vector<float, 4> {(float)_slang_vector_get_element(_S101, 0), (float)_slang_vector_get_element(_S101, 1), (float)_slang_vector_get_element(_S101, 2), (float)_slang_vector_get_element(_S101, 3)};

#line 451
        float largest_0 = -1.00000001504746622e+30f;

#line 451
        Vector<float, 4>  rejectionPlane_1 = _S102;

#line 451
        uint32_t group_0 = 0U;
        for(;;)
        {

#line 452
            if(group_0 < 6U)
            {
            }
            else
            {

#line 452
                break;
            }

#line 453
            Vector<uint32_t, 4>  _S103 = kernelContext_10->globalParams_0->r_0->meshHullInfo_0.Load(6U * kernelContext_10->globalParams_0->r_0->shapeId_0.Load(j_2) + group_0);

#line 453
            float largest_1 = largest_0;

#line 453
            Vector<float, 4>  rejectionPlane_2 = rejectionPlane_1;

#line 453
            plane_2 = 0U;
            for(;;)
            {

#line 454
                if(plane_2 < (_S103.y))
                {
                }
                else
                {

#line 454
                    break;
                }

#line 455
                Vector<float, 4>  enclosure_0 = kernelContext_10->globalParams_0->r_0->meshPlanes_0.Load(_S103.x + plane_2);
                float violation_0 = dot_0(Vector<float, 3> {enclosure_0.x, enclosure_0.y, enclosure_0.z}, _S100) + enclosure_0.w;
                if(violation_0 > largest_1)
                {

#line 457
                    largest_1 = violation_0;

#line 457
                    rejectionPlane_2 = enclosure_0;

#line 457
                }

#line 454
                plane_2 = plane_2 + 1U;

#line 454
            }

#line 452
            uint32_t _S104 = group_0 + 1U;

#line 452
            largest_0 = largest_1;

#line 452
            rejectionPlane_1 = rejectionPlane_2;

#line 452
            group_0 = _S104;

#line 452
        }

#line 460
        Vector<uint32_t, 4>  info_0 = kernelContext_10->globalParams_0->r_0->meshClusterInfo_0.Load(kernelContext_10->globalParams_0->r_0->shapeId_0.Load(i_11));
        uint32_t _S105 = info_0.x;

#line 461
        plane_2 = _S105;
        for(;;)
        {

#line 462
            if(plane_2 < (_S105 + info_0.y))
            {
            }
            else
            {

#line 462
                break;
            }

#line 462
            bool _S106 = meshClusterMayContact_0(i_11, j_2, plane_2, kernelContext_10);

            Vector<uint32_t, 4>  range_0 = kernelContext_10->globalParams_0->r_0->meshClusterRanges_0.Load(plane_2);
            if(_S106)
            {

#line 465
                uint32_t lane_0 = 0U;
                for(;;)
                {

#line 466
                    if(lane_0 < (range_0.y))
                    {
                    }
                    else
                    {

#line 466
                        break;
                    }

#line 467
                    uint32_t node_3 = kernelContext_10->globalParams_0->r_0->meshSampleIndices_0.Load(range_0.x + lane_0);

#line 467
                    Vector<float, 4>  pd_1;

#line 467
                    Vector<float, 4>  na_1;

#line 467
                    bool _S107 = nodeContact_0(i_11, j_2, node_3, &pd_1, &na_1, rejectionPlane_1, kernelContext_10);
                    if(_S107)
                    {

#line 469
                        uint32_t * _S108 = (&(kernelContext_10->globalParams_0->r_0->control_0)[int(2)]);

#line 469
                        uint32_t _S109 = *_S108;

#line 469
                        *_S108 = *_S108 + 1U;
                        if(_S109 >= ((&kernelContext_10->globalParams_0->r_0->c_0)->contactCapacity_0))
                        {

#line 470
                            fail_0(32U, kernelContext_10);

#line 470
                            return;
                        }

#line 471
                        *(&(kernelContext_10->globalParams_0->r_0->meta_0)[_S109]) = Vector<uint32_t, 4> (i_11, j_2, node_3, 0U);

#line 471
                        *(&(kernelContext_10->globalParams_0->r_0->pointDepth_0)[_S109]) = pd_1;

#line 471
                        *(&(kernelContext_10->globalParams_0->r_0->normalArea_0)[_S109]) = na_1;
                        uint32_t * _S110 = (&(kernelContext_10->globalParams_0->r_0->contactCounts_0)[pair_0]);

#line 472
                        *_S110 = *_S110 + 1U;

#line 468
                    }

#line 466
                    lane_0 = lane_0 + 1U;

#line 466
                }

#line 465
            }

#line 476
            if(_S106)
            {

#line 476
                plane_2 = plane_2 + 1U;

#line 476
            }
            else
            {

#line 476
                plane_2 = range_0.z;

#line 476
            }

#line 462
        }

#line 444
        pass_0 = pass_0 + 1U;

#line 444
    }

#line 479
    return;
}


#line 513
void _collectMeshFrontiers(void* _S111, void* entryPointParams_0, void* globalParams_1)
{

#line 513
    ComputeThreadVaryingInput * _S112 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S111));

#line 513
    KernelContext_0 kernelContext_11;

#line 513
    (&kernelContext_11)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));

#line 513
    cpuCollectMesh_0((_S112->groupID + _S112->groupThreadID).x, &kernelContext_11);

#line 513
    return;
}

// [numthreads(1, 1, 1)]
SLANG_PRELUDE_EXPORT
void collectMeshFrontiers_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _collectMeshFrontiers(varyingInput, entryPointParams, globalParams);
}
// [numthreads(1, 1, 1)]
SLANG_PRELUDE_EXPORT
void collectMeshFrontiers_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    _collectMeshFrontiers(&threadInput, entryPointParams, globalParams);
}
// [numthreads(1, 1, 1)]
SLANG_PRELUDE_EXPORT
void collectMeshFrontiers(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                collectMeshFrontiers_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}

#include <atomic>
#include <future>
#include <vector>
#include <cstring>
#include <type_traits>
#include <mutex>
#include <utility>
// Per-NativeKernel scratch: owned and freed through the C ABI, never process-global.
// Only capacities are reused. Every contact and count is reconstructed on dispatch.
struct Scratch {
        using Meta = std::remove_pointer_t<decltype(std::declval<Resources_0>().meta_0.data)>;
        using Point = std::remove_pointer_t<decltype(std::declval<Resources_0>().pointDepth_0.data)>;
        using Normal = std::remove_pointer_t<decltype(std::declval<Resources_0>().normalArea_0.data)>;
        struct Hits {
            std::vector<Meta> meta;
            std::vector<Point> point;
            std::vector<Normal> normal;
            uint32_t flags = 0, count = 0, attempts = 0;
        };
    std::vector<Hits> hits;
    std::mutex mutex;
    uint64_t stats[4] = {}; // pairs, query attempts, overflow retries, retained bytes
};
extern "C" void* cpu_scratch_create() noexcept {
    try { return new Scratch; } catch (...) { return nullptr; }
}
extern "C" void cpu_scratch_destroy(void* ptr) noexcept { delete static_cast<Scratch*>(ptr); }
extern "C" void cpu_scratch_stats(void* ptr, uint64_t* out) noexcept {
    auto& scratch = *static_cast<Scratch*>(ptr);
    std::lock_guard<std::mutex> lock(scratch.mutex);
    std::memcpy(out, scratch.stats, sizeof(scratch.stats));
}
extern "C" int cpu_dispatch(uint32_t count, void** buffers, uint64_t* bytes, double* values, void* scratch_ptr){
Resources_0 res={};
GlobalParams_0 globals={};globals.r_0=&res;
res.c_0.activeTimestep_0=values[0];
res.c_0.N_0=values[1];
res.c_0.shapes_0=values[2];
res.c_0.pairCapacity_0=values[3];
res.c_0.contactCapacity_0=values[4];
res.c_0.binCapacity_0=values[5];
res.c_0.binCount_0=values[6];
res.c_0.bins_0[0]=values[7];
res.c_0.bins_0[1]=values[8];
res.c_0.bins_0[2]=values[9];
res.c_0.hashSize_0=values[10];
res.c_0.dt_0=values[11];
res.c_0.kn_0=values[12];
res.c_0.kt_0=values[13];
res.c_0.mu_0=values[14];
res.c_0.normalViscosity_0=values[15];
res.c_0.skin_0=values[16];
res.c_0.drag_0=values[17];
res.c_0.maxOverlap_0=values[18];
res.c_0.historyStep_0=values[19];
res.c_0.meshGeometry_0=values[20];
res.c_0.periodic_0=values[21];
res.c_0.wallCount_0=values[22];
res.c_0.wallBoxEnabled_0=values[23];
res.c_0.sourceSurfaceNormal_0=values[24];
res.control_0.data=reinterpret_cast<decltype(res.control_0.data)>(buffers[0]);
res.control_0.count=bytes[0]/sizeof(*res.control_0.data);
res.cell_0.data=reinterpret_cast<decltype(res.cell_0.data)>(buffers[1]);
res.cell_0.count=bytes[1]/sizeof(*res.cell_0.data);
res.position_0.data=reinterpret_cast<decltype(res.position_0.data)>(buffers[2]);
res.position_0.count=bytes[2]/sizeof(*res.position_0.data);
res.positionRemainder_0.data=reinterpret_cast<decltype(res.positionRemainder_0.data)>(buffers[3]);
res.positionRemainder_0.count=bytes[3]/sizeof(*res.positionRemainder_0.data);
res.velocityRemainder_0.data=reinterpret_cast<decltype(res.velocityRemainder_0.data)>(buffers[4]);
res.velocityRemainder_0.count=bytes[4]/sizeof(*res.velocityRemainder_0.data);
res.momentumRemainder_0.data=reinterpret_cast<decltype(res.momentumRemainder_0.data)>(buffers[5]);
res.momentumRemainder_0.count=bytes[5]/sizeof(*res.momentumRemainder_0.data);
res.rotationRemainder_0.data=reinterpret_cast<decltype(res.rotationRemainder_0.data)>(buffers[6]);
res.rotationRemainder_0.count=bytes[6]/sizeof(*res.rotationRemainder_0.data);
res.referencePositionRemainder_0.data=reinterpret_cast<decltype(res.referencePositionRemainder_0.data)>(buffers[7]);
res.referencePositionRemainder_0.count=bytes[7]/sizeof(*res.referencePositionRemainder_0.data);
res.cellLog_0.data=reinterpret_cast<decltype(res.cellLog_0.data)>(buffers[8]);
res.cellLog_0.count=bytes[8]/sizeof(*res.cellLog_0.data);
res.accumulatedRemainder_0.data=reinterpret_cast<decltype(res.accumulatedRemainder_0.data)>(buffers[9]);
res.accumulatedRemainder_0.count=bytes[9]/sizeof(*res.accumulatedRemainder_0.data);
res.velocity_0.data=reinterpret_cast<decltype(res.velocity_0.data)>(buffers[10]);
res.velocity_0.count=bytes[10]/sizeof(*res.velocity_0.data);
res.rotation_0.data=reinterpret_cast<decltype(res.rotation_0.data)>(buffers[11]);
res.rotation_0.count=bytes[11]/sizeof(*res.rotation_0.data);
res.momentum_0.data=reinterpret_cast<decltype(res.momentum_0.data)>(buffers[12]);
res.momentum_0.count=bytes[12]/sizeof(*res.momentum_0.data);
res.force_0.data=reinterpret_cast<decltype(res.force_0.data)>(buffers[13]);
res.force_0.count=bytes[13]/sizeof(*res.force_0.data);
res.torque_0.data=reinterpret_cast<decltype(res.torque_0.data)>(buffers[14]);
res.torque_0.count=bytes[14]/sizeof(*res.torque_0.data);
res.referencePosition_0.data=reinterpret_cast<decltype(res.referencePosition_0.data)>(buffers[15]);
res.referencePosition_0.count=bytes[15]/sizeof(*res.referencePosition_0.data);
res.referenceCell_0.data=reinterpret_cast<decltype(res.referenceCell_0.data)>(buffers[16]);
res.referenceCell_0.count=bytes[16]/sizeof(*res.referenceCell_0.data);
res.shapeId_0.data=reinterpret_cast<decltype(res.shapeId_0.data)>(buffers[17]);
res.shapeId_0.count=bytes[17]/sizeof(*res.shapeId_0.data);
res.body_0.data=reinterpret_cast<decltype(res.body_0.data)>(buffers[18]);
res.body_0.count=bytes[18]/sizeof(*res.body_0.data);
res.invI_0.data=reinterpret_cast<decltype(res.invI_0.data)>(buffers[19]);
res.invI_0.count=bytes[19]/sizeof(*res.invI_0.data);
res.grids_0.data=reinterpret_cast<decltype(res.grids_0.data)>(buffers[20]);
res.grids_0.count=bytes[20]/sizeof(*res.grids_0.data);
res.sampleInfo_0.data=reinterpret_cast<decltype(res.sampleInfo_0.data)>(buffers[21]);
res.sampleInfo_0.count=bytes[21]/sizeof(*res.sampleInfo_0.data);
res.gridOrigin_0.data=reinterpret_cast<decltype(res.gridOrigin_0.data)>(buffers[22]);
res.gridOrigin_0.count=bytes[22]/sizeof(*res.gridOrigin_0.data);
res.distance_0.data=reinterpret_cast<decltype(res.distance_0.data)>(buffers[23]);
res.distance_0.count=bytes[23]/sizeof(*res.distance_0.data);
res.sample_0.data=reinterpret_cast<decltype(res.sample_0.data)>(buffers[24]);
res.sample_0.count=bytes[24]/sizeof(*res.sample_0.data);
res.meshRoots_0.data=reinterpret_cast<decltype(res.meshRoots_0.data)>(buffers[25]);
res.meshRoots_0.count=bytes[25]/sizeof(*res.meshRoots_0.data);
res.meshHullInfo_0.data=reinterpret_cast<decltype(res.meshHullInfo_0.data)>(buffers[26]);
res.meshHullInfo_0.count=bytes[26]/sizeof(*res.meshHullInfo_0.data);
res.meshPlanes_0.data=reinterpret_cast<decltype(res.meshPlanes_0.data)>(buffers[27]);
res.meshPlanes_0.count=bytes[27]/sizeof(*res.meshPlanes_0.data);
res.meshMin_0.data=reinterpret_cast<decltype(res.meshMin_0.data)>(buffers[28]);
res.meshMin_0.count=bytes[28]/sizeof(*res.meshMin_0.data);
res.meshMax_0.data=reinterpret_cast<decltype(res.meshMax_0.data)>(buffers[29]);
res.meshMax_0.count=bytes[29]/sizeof(*res.meshMax_0.data);
res.meshLinks_0.data=reinterpret_cast<decltype(res.meshLinks_0.data)>(buffers[30]);
res.meshLinks_0.count=bytes[30]/sizeof(*res.meshLinks_0.data);
res.meshTriangles_0.data=reinterpret_cast<decltype(res.meshTriangles_0.data)>(buffers[31]);
res.meshTriangles_0.count=bytes[31]/sizeof(*res.meshTriangles_0.data);
res.meshNormals_0.data=reinterpret_cast<decltype(res.meshNormals_0.data)>(buffers[32]);
res.meshNormals_0.count=bytes[32]/sizeof(*res.meshNormals_0.data);
res.meshFrontier_0.data=reinterpret_cast<decltype(res.meshFrontier_0.data)>(buffers[33]);
res.meshFrontier_0.count=bytes[33]/sizeof(*res.meshFrontier_0.data);
res.meshPairMask_0.data=reinterpret_cast<decltype(res.meshPairMask_0.data)>(buffers[34]);
res.meshPairMask_0.count=bytes[34]/sizeof(*res.meshPairMask_0.data);
res.meshPairTransform_0.data=reinterpret_cast<decltype(res.meshPairTransform_0.data)>(buffers[35]);
res.meshPairTransform_0.count=bytes[35]/sizeof(*res.meshPairTransform_0.data);
res.meshClusterInfo_0.data=reinterpret_cast<decltype(res.meshClusterInfo_0.data)>(buffers[36]);
res.meshClusterInfo_0.count=bytes[36]/sizeof(*res.meshClusterInfo_0.data);
res.meshClusterRanges_0.data=reinterpret_cast<decltype(res.meshClusterRanges_0.data)>(buffers[37]);
res.meshClusterRanges_0.count=bytes[37]/sizeof(*res.meshClusterRanges_0.data);
res.meshClusterMin_0.data=reinterpret_cast<decltype(res.meshClusterMin_0.data)>(buffers[38]);
res.meshClusterMin_0.count=bytes[38]/sizeof(*res.meshClusterMin_0.data);
res.meshClusterMax_0.data=reinterpret_cast<decltype(res.meshClusterMax_0.data)>(buffers[39]);
res.meshClusterMax_0.count=bytes[39]/sizeof(*res.meshClusterMax_0.data);
res.meshSampleIndices_0.data=reinterpret_cast<decltype(res.meshSampleIndices_0.data)>(buffers[40]);
res.meshSampleIndices_0.count=bytes[40]/sizeof(*res.meshSampleIndices_0.data);
res.binCounts_0.data=reinterpret_cast<decltype(res.binCounts_0.data)>(buffers[41]);
res.binCounts_0.count=bytes[41]/sizeof(*res.binCounts_0.data);
res.binParticles_0.data=reinterpret_cast<decltype(res.binParticles_0.data)>(buffers[42]);
res.binParticles_0.count=bytes[42]/sizeof(*res.binParticles_0.data);
res.pairCounts_0.data=reinterpret_cast<decltype(res.pairCounts_0.data)>(buffers[43]);
res.pairCounts_0.count=bytes[43]/sizeof(*res.pairCounts_0.data);
res.pairOffsets_0.data=reinterpret_cast<decltype(res.pairOffsets_0.data)>(buffers[44]);
res.pairOffsets_0.count=bytes[44]/sizeof(*res.pairOffsets_0.data);
res.pairs_0.data=reinterpret_cast<decltype(res.pairs_0.data)>(buffers[45]);
res.pairs_0.count=bytes[45]/sizeof(*res.pairs_0.data);
res.contactCounts_0.data=reinterpret_cast<decltype(res.contactCounts_0.data)>(buffers[46]);
res.contactCounts_0.count=bytes[46]/sizeof(*res.contactCounts_0.data);
res.contactOffsets_0.data=reinterpret_cast<decltype(res.contactOffsets_0.data)>(buffers[47]);
res.contactOffsets_0.count=bytes[47]/sizeof(*res.contactOffsets_0.data);
res.meta_0.data=reinterpret_cast<decltype(res.meta_0.data)>(buffers[48]);
res.meta_0.count=bytes[48]/sizeof(*res.meta_0.data);
res.pointDepth_0.data=reinterpret_cast<decltype(res.pointDepth_0.data)>(buffers[49]);
res.pointDepth_0.count=bytes[49]/sizeof(*res.pointDepth_0.data);
res.normalArea_0.data=reinterpret_cast<decltype(res.normalArea_0.data)>(buffers[50]);
res.normalArea_0.count=bytes[50]/sizeof(*res.normalArea_0.data);
res.slip_0.data=reinterpret_cast<decltype(res.slip_0.data)>(buffers[51]);
res.slip_0.count=bytes[51]/sizeof(*res.slip_0.data);
res.contactForce_0.data=reinterpret_cast<decltype(res.contactForce_0.data)>(buffers[52]);
res.contactForce_0.count=bytes[52]/sizeof(*res.contactForce_0.data);
res.normalLossPower_0.data=reinterpret_cast<decltype(res.normalLossPower_0.data)>(buffers[53]);
res.normalLossPower_0.count=bytes[53]/sizeof(*res.normalLossPower_0.data);
res.normalDamping_0.data=reinterpret_cast<decltype(res.normalDamping_0.data)>(buffers[54]);
res.normalDamping_0.count=bytes[54]/sizeof(*res.normalDamping_0.data);
res.forceVelocity_0.data=reinterpret_cast<decltype(res.forceVelocity_0.data)>(buffers[55]);
res.forceVelocity_0.count=bytes[55]/sizeof(*res.forceVelocity_0.data);
res.forceOmega_0.data=reinterpret_cast<decltype(res.forceOmega_0.data)>(buffers[56]);
res.forceOmega_0.count=bytes[56]/sizeof(*res.forceOmega_0.data);
res.oldMeta_0.data=reinterpret_cast<decltype(res.oldMeta_0.data)>(buffers[57]);
res.oldMeta_0.count=bytes[57]/sizeof(*res.oldMeta_0.data);
res.oldSlip_0.data=reinterpret_cast<decltype(res.oldSlip_0.data)>(buffers[58]);
res.oldSlip_0.count=bytes[58]/sizeof(*res.oldSlip_0.data);
res.oldNormal_0.data=reinterpret_cast<decltype(res.oldNormal_0.data)>(buffers[59]);
res.oldNormal_0.count=bytes[59]/sizeof(*res.oldNormal_0.data);
res.oldHash_0.data=reinterpret_cast<decltype(res.oldHash_0.data)>(buffers[60]);
res.oldHash_0.count=bytes[60]/sizeof(*res.oldHash_0.data);
res.hash_0.data=reinterpret_cast<decltype(res.hash_0.data)>(buffers[61]);
res.hash_0.count=bytes[61]/sizeof(*res.hash_0.data);
res.heads_0.data=reinterpret_cast<decltype(res.heads_0.data)>(buffers[62]);
res.heads_0.count=bytes[62]/sizeof(*res.heads_0.data);
res.next_0.data=reinterpret_cast<decltype(res.next_0.data)>(buffers[63]);
res.next_0.count=bytes[63]/sizeof(*res.next_0.data);
res.particleStress_0.data=reinterpret_cast<decltype(res.particleStress_0.data)>(buffers[64]);
res.particleStress_0.count=bytes[64]/sizeof(*res.particleStress_0.data);
res.timestepRows_0.data=reinterpret_cast<decltype(res.timestepRows_0.data)>(buffers[65]);
res.timestepRows_0.count=bytes[65]/sizeof(*res.timestepRows_0.data);
res.particleDiagnostic_0.data=reinterpret_cast<decltype(res.particleDiagnostic_0.data)>(buffers[66]);
res.particleDiagnostic_0.count=bytes[66]/sizeof(*res.particleDiagnostic_0.data);
res.diagnosticRows_0.data=reinterpret_cast<decltype(res.diagnosticRows_0.data)>(buffers[67]);
res.diagnosticRows_0.count=bytes[67]/sizeof(*res.diagnosticRows_0.data);
res.accumulated_0.data=reinterpret_cast<decltype(res.accumulated_0.data)>(buffers[68]);
res.accumulated_0.count=bytes[68]/sizeof(*res.accumulated_0.data);
res.fabricRows_0.data=reinterpret_cast<decltype(res.fabricRows_0.data)>(buffers[69]);
res.fabricRows_0.count=bytes[69]/sizeof(*res.fabricRows_0.data);
res.wallPlane_0.data=reinterpret_cast<decltype(res.wallPlane_0.data)>(buffers[70]);
res.wallPlane_0.count=bytes[70]/sizeof(*res.wallPlane_0.data);
res.wallRemainder_0.data=reinterpret_cast<decltype(res.wallRemainder_0.data)>(buffers[71]);
res.wallRemainder_0.count=bytes[71]/sizeof(*res.wallRemainder_0.data);
res.wallVelocity_0.data=reinterpret_cast<decltype(res.wallVelocity_0.data)>(buffers[72]);
res.wallVelocity_0.count=bytes[72]/sizeof(*res.wallVelocity_0.data);
res.wallMaterial_0.data=reinterpret_cast<decltype(res.wallMaterial_0.data)>(buffers[73]);
res.wallMaterial_0.count=bytes[73]/sizeof(*res.wallMaterial_0.data);
res.wallBoxIds_0.data=reinterpret_cast<decltype(res.wallBoxIds_0.data)>(buffers[74]);
res.wallBoxIds_0.count=bytes[74]/sizeof(*res.wallBoxIds_0.data);
res.wallRows_0.data=reinterpret_cast<decltype(res.wallRows_0.data)>(buffers[75]);
res.wallRows_0.count=bytes[75]/sizeof(*res.wallRows_0.data);
res.wallAccumulated_0.data=reinterpret_cast<decltype(res.wallAccumulated_0.data)>(buffers[76]);
res.wallAccumulated_0.count=bytes[76]/sizeof(*res.wallAccumulated_0.data);
ComputeVaryingInput vi={};vi.endGroupID={(count+0)/1,1,1};

// Inserted inside the generated C ABI bridge, after compiler-defined resource
// binding. Each pair gets private output/control storage. Geometry and state
// are read-only; contactCounts has one writer per pair. Compaction is serial in
// pair order, preserving the original collector's arithmetic and contact order.
if (uint32_t(values[25]) > 1 && count > 1 && res.control_0.data[0] == 0) {
    try {
        auto& scratch = *static_cast<Scratch*>(scratch_ptr);
        std::lock_guard<std::mutex> lock(scratch.mutex);
        auto& hits = scratch.hits;
        hits.resize(count);
        std::memset(scratch.stats, 0, sizeof(scratch.stats));
        std::atomic<uint32_t> next{0};
        const uint32_t limit = res.c_0.contactCapacity_0;
        auto worker = [&]() {
            for (;;) {
                uint32_t pair = next.fetch_add(1, std::memory_order_relaxed);
                if (pair >= count) return;
                auto& hit = hits[pair];
                hit.flags = hit.count = hit.attempts = 0;
                uint32_t capacity = uint32_t(std::min(uint64_t(limit),
                    std::max(uint64_t(1024), uint64_t(hit.meta.size()))));
                for (;;) {
                    // Never shrink a warm allocation when the global limit changes.
                    // Grow independently so a caught allocation failure cannot leave
                    // one successful allocation hiding another unfinished allocation.
                    hit.meta.resize(std::max(size_t(capacity), hit.meta.size()));
                    hit.point.resize(std::max(size_t(capacity), hit.point.size()));
                    hit.normal.resize(std::max(size_t(capacity), hit.normal.size()));
                    ++hit.attempts;
                    auto local = res;
                    uint32_t control[16];
                    std::memcpy(control, res.control_0.data, sizeof(control));
                    control[2] = 0;
                    local.control_0.data = control;
                    local.control_0.count = 16;
                    local.c_0.contactCapacity_0 = capacity;
                    local.meta_0.data = hit.meta.data(); local.meta_0.count = capacity;
                    local.pointDepth_0.data = hit.point.data(); local.pointDepth_0.count = capacity;
                    local.normalArea_0.data = hit.normal.data(); local.normalArea_0.count = capacity;
                    res.contactCounts_0.data[pair] = 0;
                    auto localGlobals = globals;
                    localGlobals.r_0 = &local;
                    auto localVi = vi;
                    localVi.startGroupID = {pair, 0, 0};
                    localVi.endGroupID = {pair + 1, 1, 1};
                    collectMeshFrontiers(&localVi, nullptr, &localGlobals);
                    if (control[0] == 32 && capacity < limit) {
                        capacity = uint32_t(std::min(uint64_t(limit), uint64_t(capacity) * 2));
                        continue;
                    }
                    hit.flags = control[0]; hit.count = control[2];
                    // Preserve allocated storage; only the valid prefix is copied.
                    break;
                }
            }
        };
        std::vector<std::future<void>> tasks;
        uint32_t workers = std::min(uint32_t(values[25]), count);
        for (uint32_t i = 0; i < workers; ++i)
            tasks.emplace_back(std::async(std::launch::async, worker));
        for (auto& task : tasks) task.get();
        uint64_t total = 0;
        uint32_t flags = 0;
        for (auto& hit : hits) {
            total += hit.count; flags |= hit.flags;
            scratch.stats[0]++;
            scratch.stats[1] += hit.attempts;
            scratch.stats[2] += hit.attempts - 1;
            scratch.stats[3] += hit.meta.capacity() * sizeof(Scratch::Meta)
                + hit.point.capacity() * sizeof(Scratch::Point)
                + hit.normal.capacity() * sizeof(Scratch::Normal);
        }
        res.control_0.data[2] = uint32_t(std::min(total, uint64_t(UINT32_MAX)));
        if (total > limit) flags |= 32;
        res.control_0.data[0] |= flags;
        if (flags) return 0; // Existing checked rollback/resize handles failure.
        size_t offset = 0;
        for (auto& hit : hits) {
            if (hit.count) {
                std::memcpy(res.meta_0.data + offset, hit.meta.data(), hit.count * sizeof(Scratch::Meta));
                std::memcpy(res.pointDepth_0.data + offset, hit.point.data(), hit.count * sizeof(Scratch::Point));
                std::memcpy(res.normalArea_0.data + offset, hit.normal.data(), hit.count * sizeof(Scratch::Normal));
            }
            offset += hit.count;
        }
        return 0;
    } catch (...) {
        // All async futures join during unwinding; no C++ exception crosses ctypes.
        return 1;
    }
}

collectMeshFrontiers(&vi,nullptr,&globals);
return 0;
}

extern "C" void cpu_profile_reset() { std::memset(profile_calls,0,sizeof(profile_calls)); std::memset(profile_ns,0,sizeof(profile_ns)); }
extern "C" void cpu_profile_read(uint64_t* calls,uint64_t* nanos) { std::memcpy(calls,profile_calls,sizeof(profile_calls)); std::memcpy(nanos,profile_ns,sizeof(profile_ns)); }
