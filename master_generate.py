import pptx
class MasterGenerator:
    def __init__(self, prs:pptx.Presentation):
        self.prs = prs
        self.masters = self.prs.slide_masters

    def build_mapping(self):
        n_cluster = len(self.masters)
        mapping = {}