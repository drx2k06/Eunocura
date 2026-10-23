from pymongo.errors import PyMongoError
from fastapi import APIRouter, Depends
from typing import Optional


from src.middlewares.auth import check_doctor_exists
from src.db import user_collection

router = APIRouter()

# Never project the credential out of the database. A patient list is a UI payload,
# not an authentication record; shipping password hashes to the browser is a leak even
# if the hash is slow to crack.
PATIENT_PROJECTION = {"hashed_password": 0}

@router.get("/")
async def get_patients(
    filter: Optional[str] = None,
    _ = Depends(check_doctor_exists)
):
    query: dict = {"role": "patient"}
    if filter:
        query["$text"] = {"$search": filter}
        cursor = user_collection.find(
            query,
            {"score": {"$meta": "textScore"}, **PATIENT_PROJECTION}
        ).sort([("score", {"$meta": "textScore"})]).limit(10)
    else:
        cursor = user_collection.find(query, PATIENT_PROJECTION).limit(10)

    results = await cursor.to_list(length=10)
    
    # Convert ObjectId to string
    for user in results:
        if "_id" in user:
            user["_id"] = str(user["_id"])
            
    return results

@router.get("/{patient_id}")
async def get_patient(patient_id: str, _ = Depends(check_doctor_exists)):
    from bson import ObjectId
    from fastapi import HTTPException
    
    try:
        try:
            query = {"_id": ObjectId(patient_id), "role": "patient"}
        except:
             query = {"_id": patient_id, "role": "patient"}
             
        user = await user_collection.find_one(query, PATIENT_PROJECTION)
        if not user:
             raise HTTPException(status_code=404, detail="Patient not found")
        
        user["_id"] = str(user["_id"])
        return user
    except (HTTPException, PyMongoError):
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))