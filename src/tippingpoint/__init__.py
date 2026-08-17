__name__ = "tippingpoint"
__author__ = "Ryan Duecker"
__version__ = "0.5.1"

from .models import MarketingReturnCurve as MarketingReturnCurve
from .portfolio import PortfolioAllocator as PortfolioAllocator
from .mmm import MultiChannelMMM as MultiChannelMMM
from .math import weibull_adstock as weibull_adstock
from .validation import validate_curve_experiments as validate_curve_experiments
from .validation import validate_multichannel_experiments as validate_multichannel_experiments
from .evaluation import evaluate_curve_fit as evaluate_curve_fit
from .evaluation import format_fit_report as format_fit_report
