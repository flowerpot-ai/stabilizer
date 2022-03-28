import torch
import torch.nn

class FGM(object):
    """
    Example
    # initialization
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in processor:
        # normal training
        loss = model(batch_input, batch_label)
        loss.backward() # backward,get the normal gradients
        # begin adversarial training
        fgm.attack() # ddd adversarial perturbation to the embedding
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # Backpropagation, and on the basis of the normal grad, accumulate the gradient of the adversarial training
        fgm.restore() # restore embedding parameters
        # gradient descent, update parameters as usual
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self, model, epsilon=1.0,emb_name='word_embeddings.'):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
        self.emb_name=emb_name

    def attack(self):
        emb_name=self.emb_name
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        emb_name=self.emb_name
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD(object):
    """
    Example
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label in processor:
        # normal training
        loss = model(batch_input, batch_label)
        loss.backward() # Backpropagation, get normal grad
        pgd.backup_grad()
        # adversarial training
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) #add adversarial perturbation to embedding, backup param.processor during first attack
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # Backpropagation, and on the basis of the normal grad, accumulate the gradient of the adversarial training
        pgd.restore() # restore embedding parameters
        # gradient descent, update parameters
        optimizer.step()
        model.zero_grad()
    """

    def __init__(self, model, emb_name, epsilon=1.0, alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FreeLB(object):
    """
    https://arxiv.org/pdf/1909.11764.pdf
    ExampleL:Currently only for bert
    freelb = FreeLB()
    K = 3
    for batch_input, batch_label in processor:
        loss = freelb.attack(model,inputs,.....)
    """

    def __init__(
        self,
        adv_K,
        adv_lr,
        adv_init_mag,
        adv_max_norm=0.0,
        adv_norm_type="l2",
        base_model="bert",
    ):
        self.adv_K = adv_K
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_init_mag = adv_init_mag
        self.adv_norm_type = adv_norm_type
        self.base_model = base_model

    def attack(self, model, inputs, gradient_accumulation_steps=1):
        input_ids = inputs["input_ids"]
        if isinstance(model, torch.nn.DataParallel):
            embeds_init = getattr(
                model.module, self.base_model
            ).embeddings.word_embeddings(input_ids)
        else:
            embeds_init = getattr(model, self.base_model).embeddings.word_embeddings(
                input_ids
            )
        if self.adv_init_mag > 0:
            input_mask = inputs["attention_mask"].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            if self.adv_norm_type == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(
                    -1, 1
                ) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.adv_norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(
                    -self.adv_init_mag, self.adv_init_mag
                )
                delta = delta * input_mask.unsqueeze(2)
        else:
            delta = torch.zeros_like(embeds_init)

        for astep in range(self.adv_K):
            delta.requires_grad_()
            inputs["inputs_embeds"] = delta + embeds_init
            inputs["input_ids"] = None
            outputs = model(**inputs)
            loss, logits = outputs[
                :2
            ]  # model outputs are always tuple in transformers (see doc)
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss = loss / gradient_accumulation_steps
            loss.backward()
            delta_grad = delta.grad.clone().detach()
            if self.adv_norm_type == "l2":
                denorm = torch.norm(
                    delta_grad.view(delta_grad.size(0), -1), dim=1
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta_norm = torch.norm(
                        delta.view(delta.size(0), -1).float(), p=2, dim=1
                    ).detach()
                    exceed_mask = (delta_norm > self.adv_max_norm).to(embeds_init)
                    reweights = (
                        self.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)
                    ).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.adv_norm_type == "linf":
                denorm = torch.norm(
                    delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")
                ).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.adv_lr * delta_grad / denorm).detach()
                if self.adv_max_norm > 0:
                    delta = torch.clamp(
                        delta, -self.adv_max_norm, self.adv_max_norm
                    ).detach()
            else:
                raise ValueError(
                    "Norm type {} not specified.".format(self.adv_norm_type)
                )
            if isinstance(model, torch.nn.DataParallel):
                embeds_init = getattr(
                    model.module, self.base_model
                ).embeddings.word_embeddings(input_ids)
            else:
                embeds_init = getattr(
                    model, self.base_model
                ).embeddings.word_embeddings(input_ids)
        return loss