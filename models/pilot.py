"""
Pilot data model.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OracleWeights:
    """
    Preference weights used by the oracle scorer.
    Must sum to 100.
    """
    tafb: int = 20          # Time Away From Base
    hotel_nights: int = 18  # Nights away from home
    report_time: int = 12   # Wake-up / report time burden
    aircraft: int = 15      # Aircraft type preference
    credit_pay: int = 25    # Credit-hour pay
    seniority: int = 10     # Seniority eligibility (always 100 if eligible)

    def total(self) -> int:
        return (self.tafb + self.hotel_nights + self.report_time
                + self.aircraft + self.credit_pay + self.seniority)

    def validate(self) -> bool:
        return self.total() == 100


@dataclass
class Pilot:
    id: int
    name: str
    age: int
    family_status: str          # e.g.