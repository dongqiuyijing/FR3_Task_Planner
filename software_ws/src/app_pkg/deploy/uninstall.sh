#!/bin/bash
# 删除桌面快捷方式
rm -f ~/.local/share/applications/armFrApp.desktop

# 删除启动脚本和日志
rm -rf ~/armFrApp

# 清除ROS工作空间编译文件
WS_PATH=$(find ~ -name "software_ws" -type d -print -quit 2>/dev/null)
if [ -n "$WS_PATH" ]; then
    rm -rf $WS_PATH/build/armFrApp
    rm -rf $WS_PATH/devel/lib/armFrApp
    rm -rf $WS_PATH/devel/share/armFrApp
fi

# 更新桌面数据库
update-desktop-database ~/.local/share/applications/

echo "卸载完成！所有相关文件已清除"