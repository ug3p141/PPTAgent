from presentation import Presentation
from multimodal import ImageLabler
# pluggable api/model
# 流程：
# 1. 读取PPT并进行解析
# 2. 将ppt中的图片提取出来，分为背景图片和实体图片，我想这一步至少应该有95+正确率
# 3. 将ppt中原有的元素首先按照其slide_layout进行一个分类，相同slide_layout之间再切分模板
# 4. figure/table/chart应该是通用的放，三个都提取成为图片统一管理
# 5. 若原本已有放置figure/table/chart的page则retrieve回来直接修改，我认为template中应该包含至少一页
#    是存在figure的，且figure/table/chart三类元素的布局相似，可以相互替换
# 6. 对于剩余的元素，对其textframe内容进行一个分类，
# (相似布局检测？相同的shape 顺序等)
# 4. 根据论文内容删除start with # and reference or citation in text.lower之后的内容
# 5.
class PPTAgent:
    def __init__(self):
        self.image_labler = ImageLabler()

    def run(self, template_path:str, paper_md:str):
        self.prs = Presentation(template_path)
        self.image_labler.label_images
