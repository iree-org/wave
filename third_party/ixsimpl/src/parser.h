/* SPDX-FileCopyrightText: 2026 ixsimpl contributors
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef IXS_PARSER_H
#define IXS_PARSER_H

#include "internal.h"

#include "node.h"

IXS_STATIC ixs_node *ixs_parse_impl(ixs_ctx *ctx, const char *input,
                                    size_t len);

#endif /* IXS_PARSER_H */
