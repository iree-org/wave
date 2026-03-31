/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_PRINT_H
#define IXS_PRINT_H

#include "internal.h"

#include "node.h"

IXS_STATIC size_t ixs_print_impl(ixs_node *expr, char *buf, size_t bufsize);
IXS_STATIC size_t ixs_print_c_impl(ixs_node *expr, char *buf, size_t bufsize);

#endif /* IXS_PRINT_H */
