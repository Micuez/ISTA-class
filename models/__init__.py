"""Model-based learnable optimizers for sparse recovery."""

from .alista import Alista
from .glista import Glista
from .lamp import Lamp
from .lfista import Lfista
from .lista import Lista
from .lista_cp import ListaCp
from .lista_cpss import ListaCpss
from .step_lista import StepLista
from .tista import Tista

__all__ = [
    "Alista",
    "Glista",
    "Lamp",
    "Lfista",
    "Lista",
    "ListaCp",
    "ListaCpss",
    "StepLista",
    "Tista",
    "build_model",
]


def build_model(
    model_name,
    A,
    num_layers,
    model_lam,
    share_W=False,
    D=None,
    alista_W=None,
    ss_q_per_layer=1.2,
    ss_maxq=13.0,
    glista_alti=5.0,
    gain_func="relu",
    tista_sigma2=None,
    lasso_lam=None,
):
    """Construct one unfolded model. Cells are added later by ``create_cell``."""
    if model_name == "lista":
        return Lista(A, num_layers, model_lam, share_W, D, name="Lista")
    if model_name == "lfista":
        return Lfista(A, num_layers, model_lam, share_W, D, name="Lfista")
    if model_name == "lamp":
        return Lamp(A, num_layers, model_lam, share_W, D, name="Lamp")
    if model_name == "step_lista":
        if model_lam != lasso_lam:
            raise ValueError("step_lista requires model_lam == lasso_lam.")
        return StepLista(A, num_layers, lasso_lam, D, name="StepLista")
    if model_name == "lista_cp":
        return ListaCp(A, num_layers, model_lam, share_W, D, name="ListaCp")
    if model_name == "lista_cpss":
        return ListaCpss(
            A,
            num_layers,
            model_lam,
            ss_q_per_layer,
            ss_maxq,
            share_W,
            D,
            name="ListaCpss",
        )
    if model_name == "alista":
        if alista_W is None:
            raise ValueError("alista requires W.npy.")
        return Alista(
            A,
            alista_W,
            num_layers,
            model_lam,
            ss_q_per_layer,
            ss_maxq,
            D,
            name="Alista",
        )
    if model_name == "glista":
        return Glista(
            A,
            num_layers,
            model_lam,
            ss_q_per_layer,
            ss_maxq,
            share_W,
            D,
            alti=glista_alti,
            gain_func=gain_func,
            name="Glista",
        )
    if model_name == "tista":
        return Tista(
            A,
            num_layers,
            model_lam,
            tista_sigma2,
            share_W,
            D,
            name="Tista",
        )
    raise NotImplementedError(f"Unknown model {model_name!r}.")
