# -*- coding: utf-8 -*-
"""Entry point. 程序入口。

JackFramework owns the run; this file only hands it the user interface.
JackFramework 负责整个运行流程，本文件只是把用户接口交给它。
"""

import JackFramework as jf

from UserModelImplementation.user_interface import UserInterface


def main() -> None:
    """Start the application. 启动应用。"""
    app = jf.Application(UserInterface(), 'Template-jf')
    app.start()


if __name__ == '__main__':
    main()
