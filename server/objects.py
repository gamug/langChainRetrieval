from pydantic import BaseModel


class Documento(BaseModel):
    contenido: str
    
class Pregunta(BaseModel):
    pregunta: str