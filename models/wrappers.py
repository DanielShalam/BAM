import torch
import torch.nn as nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, proj_head, pred_head=None, nested_input=True):
        super(MultiCropWrapper, self).__init__()

        self.nested_input = nested_input  # mainly for adapting to DINO-V2 sequence packing
        self.backbone = backbone
        self.proj_head = proj_head if proj_head is not None else nn.Identity()
        self.pred_head = pred_head if pred_head is not None else nn.Identity()

    def forward(self, x, withhead: bool = True, **kwargs):
        if not self.nested_input:
            # make input for DINO-V2 sequence packing
            if isinstance(x, list):
                idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]),
                                                                  return_counts=True, )[1], 0)

                if len(idx_crops) == 1:  # only 1 size
                    output = self.backbone(x[0])
                else:
                    x_concat_list, start_idx = [], 0
                    for i, end_idx in enumerate(idx_crops):
                        x_concat_list.append(torch.cat(x[start_idx: end_idx]))
                        start_idx = end_idx
                    output = self.backbone(x_concat_list)
            else:
                output = self.backbone(x)
        else:
            idx_crops = torch.cumsum(torch.unique_consecutive(torch.tensor([inp.shape[-1] for inp in x]),
                                                              return_counts=True, )[1], 0)

            start_idx, output = 0, torch.empty(0).to(x[0].device)
            for i, end_idx in enumerate(idx_crops):
                _out = self.backbone(torch.cat(x[start_idx: end_idx]), **kwargs)
                # accumulate outputs
                output = torch.cat((output, _out))
                start_idx = end_idx

        # Run the head forward on the concatenated features.
        if withhead:
            return self.pred_head(self.proj_head(output))

        return output

    def forward_head(self, x: torch.Tensor, normalize=False):
        if normalize:
            return nn.functional.normalize(self.proj_head(x), dim=-1)
        return self.proj_head(x)

    def forward_predict(self, x: torch.Tensor):
        return self.pred_head(x)


class StudentTeacherWrapper(nn.Module):
    """ Wrapping a model with a Momentum teacher
    """

    def __init__(self, student, teacher, m_schedule):
        super(StudentTeacherWrapper, self).__init__()
        self.student = student
        self.teacher = teacher
        self.m_schedule = m_schedule

        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def update_teacher(self, m: float = None):
        """ Update the momentum encoder
        """
        assert self.student.training
        if m is None:
            m = self.m_schedule[0]
            self.m_schedule = self.m_schedule[1:]

        for param_b, param_m in zip(self.student.parameters(), self.teacher.parameters()):
            param_m.data.mul_(m).add_((1 - m) * param_b.detach().data)

    # def train(self):    # TODO: teacher on eval ?
    #     super().train()
    #     self.teacher.eval()

    def forward(self, x: torch.Tensor, teacher_x: torch.Tensor = None, **forward_kwargs):
        student_out = self.student(x, **forward_kwargs)
        if not self.training:
            return student_out

        with torch.no_grad():
            # update teacher parameters
            self.update_teacher()
            teacher_out = self.teacher(x if teacher_x is None else teacher_x, **forward_kwargs)
        return student_out, teacher_out
