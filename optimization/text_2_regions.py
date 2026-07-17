from google import genai
from google.genai import types
import os


class ContactExtractor:
    def __init__(self, api_key, model_name="gemini-3-flash-preview"):
        """
        初始化 Gemini 客户端
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name


        # ======================================================
        # 修改点：使用了新的、更严格的 System Prompt，直接对应你的详细字典
        # ======================================================
        self.system_instruction = """
You are a helpful assistant. You will follow ALL rules and directions entirely and precisely.

Given a description of Person 1 and Person 2 who are physically in contact with each other, create a Markdown table with the following columns:
1. **Person 1 Body Part**
2. **Person 2 Body Part**
3. **Contact Area** (Size of the contact region)
4. **Normal Alignment** (How strictly the surface normals should align)
5. **Role Ratio (P1:P2)** (New Column!)

### 1. ALLOWED BODY PARTS
You strictly limit your output to the exact strings listed below. Do not use any other words.
If the description mentions a specific part (e.g., "calf"), map it to the corresponding part in this list (e.g., "leg").

**General:**
"head", "neck", "chest", "stomach", "butt", "back"

**Limbs (do not specify Left/Right if not sure):**
"left hand (palm)", "right hand (palm)"
"left hand (back)", "right hand (back)"
"left arm", "right arm"
"left leg", "right leg"
"left foot", "right foot"

**Waist (Choose the most specific one):**
"waist", "waist (front)", "waist (back)"

**Shoulders (Must specify Left/Right and optionally Front/Back):**
"left shoulder", "left shoulder (front)", "left shoulder (back)"
"right shoulder", "right shoulder (front)", "right shoulder (back)"

### 2. ALLOWED CONTACT ATTRIBUTES
For the "Contact Area" and "Normal Alignment" columns, you must use ONLY the specific values defined below based on the physics of the interaction.

**Column: Contact Area**
Choose one based on the size/nature of the contact:
* `point`: Very small area (e.g., fingertip poking, tapping, clicking).
* `flat`: Medium area, planar contact (e.g., palm resting, high-five, leaning against a wall).
* `wrap`: Large area, curling/grasping (e.g., holding a hand, grabbing an arm, holding a handle).
* `hug`: Very large area, full body contact (e.g., hugging, carrying on back).

### 3. ROLE RATIO GUIDELINES
Specify the ratio of "Active Movement" between Person 1 and Person 2 (format "X:Y").
* **5:5** : Symmetric interaction (e.g., Handshake, High-five, Hug). Both move equally.
* **9:1** : P1 is the attacker/active doer; P2 is static/passive (e.g., P1 punches P2, P1 touches P2's shoulder). P1 moves to P2; P2 barely moves to meet P1.
* **1:9** : P2 is active; P1 is passive.
* **7:3** : P1 is mostly active, but P2 reacts slightly (e.g., P1 attacks, P2 blocks).

### RULES
1. **Precision:** If the description implies a specific side (front/back) for the waist or shoulder, you MUST use the specific term.
2. **Left/Right:** Infer the left or right side based on the description and the orientation of the people, you SHOULD NOT specify left/right if the description is ambiguous.
3. **Exact Match:** The content in the table cells must match the allowed strings exactly.
4. **Inference:**
    * If P1 grabs P2's arm: Area=`wrap`
    * If P1 pats P2's back: Area=`flat`
    * If P1 pokes P2's cheek: Area=`point`
5. **Format:**
    * If a contact involves multiple possibilities that can't be distinguished, use a comma-separated list within the cell.
    * If there are no contacts, output the table headers only, for example, someone dodge, you do not need to output contacts.

Write the Markdown table.
"""

    def _parse_markdown_table(self, text):
        """
        内部辅助函数：解析 Markdown 表格文本为字典列表
        """
        data = []
        lines = text.split('\n')
        headers = []
        in_table = False

        for line in lines:
            line = line.strip()
            # 过滤非表格行
            if not line.startswith('|') or not line.endswith('|'):
                # 如果已经开始读表了，遇到非表格行通常意味着表格结束
                if in_table: break
                continue

            # 分割并清洗单元格
            cells = [c.strip() for c in line.split('|')]

            # 移除 split 产生的首尾空字符串 (因为 |content| Split 后首尾是空)
            if len(cells) > 0 and cells[0] == '': cells.pop(0)
            if len(cells) > 0 and cells[-1] == '': cells.pop()

            # 跳过分隔线行 (---|---)
            if '---' in line: continue

            # 提取表头
            if not headers:
                # 转小写方便后续处理 (e.g., "Contact Area" -> "contact area")
                headers = [h.lower() for h in cells]
                in_table = True
                continue

            # 提取数据
            if len(cells) == len(headers):
                row_dict = {}
                for i, header in enumerate(headers):
                    row_dict[header] = cells[i]
                data.append(row_dict)

        return data

    def map_contacts_to_ids(self, contacts, candidates_dict):
        """
        将文本格式的接触列表转换为具体的 ID 列表，并提取接触属性。

        Args:
            contacts (list): _parse_markdown_table 返回的字典列表
            candidates_dict (dict): 身体部位到 ID 列表的映射字典

        Returns:
            list: 包含 ID 数据和物理属性的结构化列表
        """
        mapped_results = []

        for item in contacts:
            # 1. 获取身体部位 (Key)
            p1_key = item.get('person 1 body part', '').lower().strip()
            p2_key = item.get('person 2 body part', '').lower().strip()

            # 2. 获取接触属性 (新增部分)
            # 对应 Prompt 中的 "Contact Area" 列 -> 映射为 contact_type (point, flat, wrap, hug)
            c_area = item.get('contact area', 'flat').lower().strip()
            # 对应 Prompt 中的 "Normal Alignment" 列 -> 映射为 align_type (strict, adaptive, none)
            c_align = item.get('normal alignment', 'strict').lower().strip()

            raw_ratio = item.get('role ratio (p1:p2)', '5:5').strip()  # 获取字符串

            try:
                # 解析 "9:1" -> [9.0, 1.0]
                parts = raw_ratio.split(':')
                if len(parts) == 2:
                    w1 = float(parts[0])
                    w2 = float(parts[1])
                else:
                    w1, w2 = 5.0, 5.0
            except:
                w1, w2 = 5.0, 5.0

            # 归一化，防止总 Loss 变大
            total_w = w1 + w2 + 1e-6
            w_p1 = w1 / total_w  # e.g., 0.9
            w_p2 = w2 / total_w  # e.g., 0.1

            # 3. 处理逗号分隔 (如果有)
            p1_keys = [k.strip() for k in p1_key.split(',')]
            p2_keys = [k.strip() for k in p2_key.split(',')]

            p1_ids = []
            p2_ids = []

            # 4. 查找 ID
            for k in p1_keys:
                if k in candidates_dict:
                    p1_ids.extend(candidates_dict[k])
                else:
                    print(f"[Warning] Key '{k}' not found in CANDIDATES for Person 1")

            for k in p2_keys:
                if k in candidates_dict:
                    p2_ids.extend(candidates_dict[k])
                else:
                    print(f"[Warning] Key '{k}' not found in CANDIDATES for Person 2")

            # 5. 构建结果
            if p1_ids and p2_ids:
                mapped_results.append({
                    'p1_part': p1_key,
                    'p1_ids': sorted(list(set(p1_ids))),  # 去重
                    'p2_part': p2_key,
                    'p2_ids': sorted(list(set(p2_ids))),  # 去重

                    # --- 新增字段 ---
                    'contact_type': c_area,  # 用于决定 top_k
                    'align_type': c_align,  # 用于决定 alignment weight
                    'w_p1': w_p1,  # [新增] P1 主动性权重
                    'w_p2': w_p2  # [新增] P2 主动性权重
                })

        return mapped_results

    def get_contacts(self, description, candidates_dict):
        """
        主函数：传入描述，返回解析后的接触列表
        Args:
            description: 文本描述
            candidates_dict: 你的 region mapping 字典
        """
        try:
            # 1. 调用 LLM
            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction),
                contents=description
            )

            # 2. 解析 Markdown 表格
            raw_contacts = self._parse_markdown_table(response.text)

            # 3. 映射 ID 和属性
            structured_contacts = self.map_contacts_to_ids(raw_contacts, candidates_dict)

            return structured_contacts

        except Exception as e:
            print(f"Error calling Gemini or Parsing: {e}")
            # 出错时返回空列表，避免程序崩溃
            return []


# ==========================================
# 3. 使用示例
# ==========================================

if __name__ == "__main__":
    # 请填入你的 API Key
    MY_API_KEY = "AIzaSyBasqVApLHbY8tRevAnU9EnraMJvpBuW0s"

    # 1. 实例化
    extractor = ContactExtractor(api_key=MY_API_KEY)

    CANDIDATES = {
        'left hand (palm)': [17, 33],
        'left hand (back)': [3, 72],
        'right hand (palm)': [23, 26],
        'right hand (back)': [27, 40],
        'left arm': [5, 11, 17, 34, 49, 57, 69],
        'right arm': [6, 8, 24, 32, 37, 44],
        'left foot': [1, 67],
        'right foot': [42, 65],
        'left leg': [15, 20, 21, 25, 30, 45, 53, 56, 60, 64],
        'right leg': [12, 16, 18, 19, 22, 28, 36, 46, 55, 62],
        'back': [13, 35, 38, 41, 63, 70, 2, 29, 10, 48, 14, 43],
        'head': [39, 51, 52, 66],
        'neck': [14, 43, 47, 54],
        'butt': [9, 71],
        'waist': [10, 48, 50, 68, 74],
        'waist (back)': [10, 48],
        'waist (front)': [50, 68, 74],
        'left shoulder (front)': [73],
        'left shoulder (back)': [2],
        'right shoulder (front)': [31],
        'right shoulder (back)': [29],
        'left shoulder': [2, 73],
        'right shoulder': [29, 31],
        'chest': [7, 58, 59, 61],
        'stomach': [0, 4, 68, 74],
    }

    # 2. 准备输入描述
    # 这是一个很好的测试用例，包含了 "right calf" (应该映射为 right leg) 和方位逻辑
    text_input = "The first person stands on the right side of the second person, facing him, and kicks the second person's right calf with their right foot."

    # 3. 获取结果
    print(f"正在分析: {text_input}")
    print("-" * 30)
    contacts = extractor.get_contacts(text_input, CANDIDATES)

    # 4. 输出结果
    # 这里直接输出，不再进行任何 Python 侧的 Left/Right 处理
    print(contacts)