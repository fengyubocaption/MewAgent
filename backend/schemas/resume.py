from pydantic import BaseModel
from typing import List, Optional


class ResumeInfo(BaseModel):
    id: int
    filename: str
    structured_data: Optional[dict] = None
    created_at: str
    updated_at: str


class ResumeListResponse(BaseModel):
    resumes: List[ResumeInfo]


class ResumeDetailResponse(BaseModel):
    id: int
    filename: str
    raw_text: str
    structured_data: Optional[dict] = None
    created_at: str
    updated_at: str


class ResumeUploadResponse(BaseModel):
    id: int
    filename: str
    message: str


class ResumeDeleteResponse(BaseModel):
    id: int
    message: str
