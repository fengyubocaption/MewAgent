from pydantic import BaseModel
from typing import List, Optional


class JDCreateRequest(BaseModel):
    title: str = ""
    company: str = ""
    jd_text: str


class JDInfo(BaseModel):
    id: int
    title: str
    company: str
    structured_data: Optional[dict] = None
    created_at: str
    updated_at: str


class JDListResponse(BaseModel):
    job_descriptions: List[JDInfo]


class JDDetailResponse(BaseModel):
    id: int
    title: str
    company: str
    raw_text: str
    structured_data: Optional[dict] = None
    created_at: str
    updated_at: str


class JDCreateResponse(BaseModel):
    id: int
    title: str
    company: str
    message: str


class JDDeleteResponse(BaseModel):
    id: int
    message: str
