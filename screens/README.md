# Count Change 截图目录

本目录用于存放数量变化检测的测试截图。

## 目录结构

```
screens/
├── sample/              # 示例截图
│   ├── count_before.png # 操作前截图（占位符）
│   └── count_after.png  # 操作后截图（占位符）
├── Android/
│   └── sample/          # Android平台示例截图（占位符）
│       ├── like_before.png
│       ├── like_after.png
│       ├── comment_before.png
│       ├── comment_after.png
│       ├── cart_before.png
│       └── cart_after.png
└── HarmonyOS/
    └── sample/          # HarmonyOS平台示例截图（占位符）
        ├── like_before.png
        ├── like_after.png
        ├── comment_before.png
        ├── comment_after.png
        ├── follow_before.png
        └── follow_after.png
```

## 截图要求

1. **格式**：支持 PNG、JPG 等常见图片格式
2. **命名**：使用 `{场景}_{before/after}.{扩展名}` 格式
3. **内容**：
   - `before` 截图：操作执行前的页面状态
   - `after` 截图：操作执行后的页面状态
   - 两张截图应该来自同一个操作流程，只是操作前后的状态不同

## 注意事项

- 当前目录中的文件都是占位符，需要替换为实际的截图文件
- 截图路径应该与 `jsons/` 目录中的JSON文件中的路径一致
- 建议使用相对路径，相对于 `jsons/` 目录
