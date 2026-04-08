import base64
import mimetypes

SYSTEM_PROMPT_MINDMAP = """
你是一个精确的数据输入助手，能够将思维导图图片转录成结构化文本。
#任务
请看下面这张思维导图。提取所有文本节点并使用缩进表示它们的层次结构。
#格式规则（严格遵循）
1. 对每个节点使用连字符“-”。
2. 使用4个空格作为缩进来表示层次结构（父子关系）。
- 根主题：无缩进
- 第一级：    -主题
- 第二级：        -副标题
3. 不要添加任何会话文本。只输出图片中的文本。
4. 如果一个节点包含多行文本，请使用空格将它们连接成一行。
5. 准确地抄写图片中的内容。
"""

def encode_image(image_path):
    """将图片转换为 Base64 格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_single_image(vl_client, image_path):
    """调用 ERNIE-4.5-VL 处理单张图片"""
    # 1. 编码图片
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 2. 猜测 MIME 类型
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/png"

    try:
        # 3. 构造请求
        response = vl_client.chat.completions.create(
            model="ernie-4.5-vl-28b-a3b", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_MINDMAP},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请按照系统提示，将这张思维导图图片转录为结构化的文本格式。"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0,
            top_p=0.8,
            extra_body={ 
                "stop": [], 
                "enable_thinking": True, 
                "frequency_penalty": 0, 
                "presence_penalty": 0, 
                "repetition_penalty": 1
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"思维导图识别失败: {e}"