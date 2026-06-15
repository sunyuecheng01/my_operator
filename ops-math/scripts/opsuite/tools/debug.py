#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import logging
import subprocess
from utils import script

logger = logging.getLogger(__name__)


def run(options, cmd_name):
    # 对构建选项做去重处理
    unique_build_ops = list(dict.fromkeys(options))
    cmd_with_opt = cmd_name + unique_build_ops
    try:
        logger.info(f"正在执行 %s", " ".join(cmd_with_opt))
        result = subprocess.run(cmd_with_opt, capture_output=False, text=True)
        if result.returncode != 0:
            logger.error(f"执行%s 失败，请查看日志\n", " ".join(cmd_with_opt))
            return
    except Exception as e:
        logger.error(f"执行 %s 失败： {e}", " ".join(cmd_with_opt))
        return
