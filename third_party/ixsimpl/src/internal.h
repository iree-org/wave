/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_INTERNAL_H
#define IXS_INTERNAL_H

/*
 * IXS_STATIC expands to `static` in the amalgamated single-TU build,
 * hiding every internal symbol.  In the normal multi-TU build it
 * expands to nothing so that cross-file linkage works as usual.
 */
#ifdef IXS_AMALGAMATED
#define IXS_STATIC static
#else
#define IXS_STATIC
#endif

#endif /* IXS_INTERNAL_H */
