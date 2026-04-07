# Count Change 测试用例说明

本目录包含数量变化检测的测试用例。

## 目录结构

```
count_change/
├── jsons/
│   ├── sample.json              # 示例测试用例
│   ├── Android/                 # Android平台测试用例
│   │   ├── sample_like.json     # 点赞数变化测试（占位符）
│   │   ├── sample_comment.json  # 评论数变化测试（占位符）
│   │   └── sample_cart.json     # 购物车数量变化测试（占位符）
│   └── HarmonyOS/               # HarmonyOS平台测试用例
│       ├── sample_like.json      # 点赞数变化测试（占位符）
│       ├── sample_comment.json   # 评论数变化测试（占位符）
│       └── sample_follow.json    # 关注数变化测试（占位符）
└── screens/                      # 截图目录（占位符）
    ├── sample/
    ├── Android/sample/
    └── HarmonyOS/sample/
```

## JSON字段说明

- `type`: 测试类型，固定为 `"count_change"`
- `screenshot_a`: 操作前的截图路径（相对路径）
- `screenshot_b`: 操作后的截图路径（相对路径）
- `bounds`: 目标控件的bounds `[left, top, right, bottom]`（占位符为 `[0, 0, 0, 0]`）
- `expected_passed`: **人工标注的groundtruth**（必需）
  - `true`: 表示人工判断应该检测到数量变化
  - `false`: 表示人工判断不应该检测到数量变化
  - **注意**：这是人工标注的期望结果，不是模型判断的结果
- `label`: 标签，`"pass"` 或 `"fail"`（向后兼容，可选）
- `description`: 测试用例描述（可选）
- `note`: 备注说明（可选）

## 使用说明

1. **替换占位符数据**：
   - 将 `bounds` 替换为实际的目标控件坐标
   - 将截图路径替换为实际的截图文件路径
   - 确保截图文件存在于 `screens/` 目录下

2. **运行测试**：
   ```python
   from aichecker import check_count_change
   import json
   
   with open("testcase/count_change/jsons/sample.json", "r") as f:
       payload = json.load(f)
   
   result = check_count_change(payload)
   print(f"检测结果: {result.passed}")
   ```

## 测试场景

### 场景1：点赞数变化
检测点击点赞按钮后，点赞数是否增加。

### 场景2：评论数变化
检测点击评论按钮后，评论数是否变化。

### 场景3：购物车数量变化
检测添加商品到购物车后，购物车商品数量是否增加。

### 场景4：关注数变化
检测点击关注按钮后，关注数是否变化。

## 注意事项

- **Groundtruth标注**：
  - `expected_passed` 必须由人工标注，不能依赖模型判断
  - 标注人员需要查看截图，判断操作后数量是否真的发生了变化
  - 这是评估模型准确性的标准答案（groundtruth）
- **占位符数据**：
  - 当前测试用例中的 `bounds` 和截图路径都是占位符，需要替换为实际数据
  - 截图文件需要放在对应的 `screens/` 子目录下
- **测试流程**：
  1. 人工标注 `expected_passed`（groundtruth）
  2. 模型检测数量变化，返回 `passed`（模型判断）
  3. 测试脚本比较两者，验证模型准确性
