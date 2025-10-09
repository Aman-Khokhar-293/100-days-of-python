#  Day 80  – Building APIs with FastAPI
# Mini Project: Bookstore API with CRUD operations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(
    title="📚 Bookstore API",
    description="A simple FastAPI project for managing books with CRUD operations",
    version="1.0.0"
)

# ----- Data Models -----
class Book(BaseModel):
    id: int
    title: str
    author: str
    price: float
    in_stock: Optional[bool] = True


# ----- In-memory Database -----
books_db: List[Book] = [
    Book(id=1, title="Atomic Habits", author="James Clear", price=15.99, in_stock=True),
    Book(id=2, title="Deep Work", author="Cal Newport", price=12.50, in_stock=True),
]


# ----- API Routes -----

@app.get("/")
def home():
    return {"message": "Welcome to the Bookstore API 📚"}


@app.get("/books", response_model=List[Book])
def get_books():
    """Get a list of all books"""
    return books_db


@app.get("/books/{book_id}", response_model=Book)
def get_book(book_id: int):
    """Get a single book by its ID"""
    for book in books_db:
        if book.id == book_id:
            return book
    raise HTTPException(status_code=404, detail="Book not found")


@app.post("/books", response_model=Book)
def add_book(new_book: Book):
    """Add a new book"""
    for book in books_db:
        if book.id == new_book.id:
            raise HTTPException(status_code=400, detail="Book with this ID already exists")
    books_db.append(new_book)
    return new_book


@app.put("/books/{book_id}", response_model=Book)
def update_book(book_id: int, updated_book: Book):
    """Update book details"""
    for index, book in enumerate(books_db):
        if book.id == book_id:
            books_db[index] = updated_book
            return updated_book
    raise HTTPException(status_code=404, detail="Book not found")


@app.delete("/books/{book_id}")
def delete_book(book_id: int):
    """Delete a book by its ID"""
    for index, book in enumerate(books_db):
        if book.id == book_id:
            deleted_book = books_db.pop(index)
            return {"message": f"Book '{deleted_book.title}' deleted successfully"}
    raise HTTPException(status_code=404, detail="Book not found")


# ----- Run the Server -----
# Use: uvicorn Day_80:app --reload
# Example: http://127.0.0.1:8000/books
# Docs: http://127.0.0.1:8000/docs
# Redoc: http://127.0.0.1:8000/redoc

