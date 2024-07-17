from presentation import Presentation
import logging


class MasterGenerator:
    # 至多八个layout
    def __init__(self, prs: Presentation, max_templates=8):
        self.prs = prs
        layout_mapping = {
            layout.name: [
                slide for slide in prs.slides if slide.slide_layout_name == layout.name
            ]
            for layout in prs.slide_layouts
        }
        if max_templates < len(layout_mapping):
            logging.warning(
                f"max_templates is less than the number of available layouts, using all available layouts"
            )
            max_templates = len(layout_mapping)
        if max_templates > len(prs.slides):
            logging.warning(
                f"max_templates is greater than the number of available slides, using all available slides"
            )
            max_templates = len(prs.slides)

        while len(layout_mapping) > max_templates:
            # while
            calc_layout()

    def build_mapping(self):
        pass


if __name__ == "__main__":
    prs = Presentation("./resource/软件所党员组织关系专项整治工作部署.pptx")
    mg = MasterGenerator(prs)
    mg.build_mapping()
