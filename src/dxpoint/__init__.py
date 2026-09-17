__name__ = "dxpoint"
__author__ = "Ryan Duecker"
__version__ = "0.5.2"

from .models import MarketingReturnCurve as MarketingReturnCurve
from .portfolio import PortfolioAllocator as PortfolioAllocator
from .multichannel import MultiChannelModel as MultiChannelModel
from .multichannel import MultiChannelMMM as MultiChannelMMM
from .math import weibull_adstock as weibull_adstock
from .validation import validate_curve_experiments as validate_curve_experiments
from .validation import validate_multichannel_experiments as validate_multichannel_experiments
from .evaluation import evaluate_curve_fit as evaluate_curve_fit
from .evaluation import format_fit_report as format_fit_report
from .capacity import CapacityProjector as CapacityProjector
from .capacity import ProjectedReturnCurve as ProjectedReturnCurve
from .capacity import CapacityMultipliers as CapacityMultipliers
