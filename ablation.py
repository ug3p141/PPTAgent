import random
import tempfile
from copy import deepcopy

from apis import API_TYPES, CodeExecutor
from llms import Role
from pptgen import PPTCrew
from presentation import GroupShape, ShapeElement, SlidePage, TextFrame
from utils import get_slide_content


class PPTCrew_wo_Decoupling(PPTCrew):
    roles: list[str] = ["agent", "typographer"]

    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ) -> SlidePage:
        temp_dir = tempfile.TemporaryDirectory()
        schema = template["content_schema"]
        edit_actions = self.staffs["agent"](
            schema=schema,
            api_docs=code_executor.get_apis_docs(API_TYPES.Agent.value),
            edit_target=self.presentation.slides[template["template_id"] - 1].to_html(),
            outline=self.simple_outline,
            metadata=self.metadata,
            text=slide_content,
            images_info=image_info,
        )
        for error_idx in range(self.retry_times):
            edited_slide: SlidePage = deepcopy(
                self.presentation.slides[template["template_id"] - 1]
            )
            feedback = code_executor.execute_actions(edit_actions, edited_slide)
            if feedback is None:
                return edited_slide
            if error_idx == self.retry_times - 1:
                raise Exception(
                    f"Failed to generate slide, tried too many times at editing\ntraceback: {feedback[1]}"
                )
            edit_actions = self.staffs["agent"].retry(*feedback, error_idx + 1)
        edited_image = self._build_slide(edited_slide, temp_dir.name)
        return self.style_adjusting(edited_slide, edited_image, code_executor)


class PPTCrew_wo_SchemaInduction(PPTCrew):
    def _hire_staffs(self, record_cost: bool, **kwargs) -> dict[str, Role]:
        new_editor = "editor_wo_schema"
        self.roles.append(new_editor)
        super()._hire_staffs(record_cost, **kwargs)
        self.staffs["editor"] = self.staffs.pop(new_editor)

    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        images_info: str,
    ) -> SlidePage:
        temp_dir = tempfile.TemporaryDirectory()
        content_schema = template["content_schema"]
        new_schema = {k: v["data"] for k, v in enumerate(content_schema.values())}
        old_data = self._prepare_schema(new_schema)
        editor_output = self.staffs["editor"](
            schema=old_data,
            outline=self.simple_outline,
            metadata=self.metadata,
            text=slide_content,
            images_info=images_info,
        )
        new_editor_output = {
            k: v["data"] for k, v in zip(content_schema.keys(), editor_output)
        }
        command_list = self._generate_commands(
            new_editor_output, content_schema, old_data
        )

        edit_actions = self.staffs["coder"](
            api_docs=code_executor.get_apis_docs(API_TYPES.Agent.value),
            edit_target=self.presentation.slides[template["template_id"] - 1].to_html(),
            command_list="\n".join([str(i) for i in command_list]),
        )
        for error_idx in range(self.retry_times):
            edited_slide: SlidePage = deepcopy(
                self.presentation.slides[template["template_id"] - 1]
            )
            feedback = code_executor.execute_actions(edit_actions, edited_slide)
            if feedback is None:
                break
            if error_idx == self.retry_times - 1:
                raise Exception(
                    f"Failed to generate slide, tried too many times at editing\ntraceback: {feedback[1]}"
                )
            edit_actions = self.staffs["coder"].retry(*feedback, error_idx + 1)
        edited_image = self._build_slide(edited_slide, temp_dir.name)
        return self.style_adjusting(edited_slide, edited_image, code_executor)


# random layout
class PPTCrew_wo_LayoutInduction(PPTCrew):
    def _generate_slide(self, slide_data, code_executor: CodeExecutor) -> SlidePage:
        slide_idx, (slide_title, slide) = slide_data
        images_info = "No Images"
        if any(
            [
                i in slide["layout"]
                for i in ["picture", "chart", "table", "diagram", "freeform"]
            ]
        ):
            images_info = self.image_information
        slide_content = f"Slide-{slide_idx+1} " + get_slide_content(
            self.doc_json, slide_title, slide
        )
        try:
            return self.synergize(
                deepcopy(self.slide_induction[random.choice(self.layout_names)]),
                slide_content,
                code_executor,
                images_info,
            )
        except Exception as e:
            print(f"generate slide {slide_idx} failed: {e}")
            return None


def monkeypatch_render():
    for cls in [
        ShapeElement,
        GroupShape,
        SlidePage,
        TextFrame,
    ]:
        cls.to_html = lambda s: s.to_pptc()


class PPTCrew_wo_HTML(PPTCrew):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        monkeypatch_render()

    def _hire_staffs(self, record_cost: bool, **kwargs) -> dict[str, Role]:
        new_coder = "coder_wo_html"
        self.roles.append(new_coder)
        super()._hire_staffs(record_cost, **kwargs)
        self.staffs["coder"] = self.staffs.pop(new_coder)
