#!/bin/bash
# 定义资源路径
DEPLOY_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) #商标图片路径
ICON_DEST="$HOME/.local/share/icons/armFrApp.png"
# echo "$DEPLOY_DIR"
# 检查依赖项
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "缺失依赖项: $1"
        exit 1
    fi
}

check_dependency roslaunch
check_dependency python3
check_dependency xdg-icon-resource

# 安装图标
echo "正在安装图标..."
mkdir -p ~/.local/share/icons

# 从部署目录直接复制图标
if ! cp "$DEPLOY_DIR/icons/app_icon.png" "$ICON_DEST"; then
    echo "图标复制失败，请检查路径权限"
    exit 1
fi

# 注册系统图标
if ! xdg-icon-resource install --size 64 "$ICON_DEST" --novendor; then
    echo "图标注册失败，请检查文件格式是否为PNG"
    exit 1
fi

# 创建桌面入口
echo "创建桌面快捷方式..."
DESKTOP_PATH="$HOME/.local/share/applications/armFrApp.desktop"
cat > "$DESKTOP_PATH" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Software
Comment=ROS Integrated Control Software
Exec=$DEPLOY_DIR/start_document.sh
Icon=armFrApp
Categories=Utility;Application;
Terminal=false
EOF

# 更新桌面数据库
if ! update-desktop-database ~/.local/share/applications/; then
    echo "桌面数据库更新失败"
    exit 1
fi

echo "安装成功！可在应用菜单中找到启动图标"