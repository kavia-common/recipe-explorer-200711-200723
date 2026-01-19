import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

OPENAPI_TAGS = [
    {"name": "Health", "description": "Service health and diagnostics."},
    {"name": "Categories", "description": "Browse recipe categories."},
    {"name": "Recipes", "description": "Browse and view recipes."},
    {"name": "Search", "description": "Search recipes by ingredients."},
    {"name": "Ratings", "description": "Create and view recipe ratings."},
]


def _normalize_database_url(raw: str) -> str:
    """
    Normalize DATABASE_URL into a SQLAlchemy-compatible URL.

    Supports:
    - postgresql://...
    - postgres://... (mapped to postgresql+psycopg://...)
    - postgresql+psycopg://... (already OK)
    """
    url = raw.strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        # Prefer psycopg v3 dialect explicitly for SQLAlchemy 2.x
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def _build_engine() -> Engine:
    """
    Create SQLAlchemy engine from DATABASE_URL.

    NOTE: The DATABASE_URL env var must be set by deployment/orchestration.
    For local dev, set DATABASE_URL to the same string found in
    database_postgresql/db_connection.txt (without the leading `psql `).
    """
    raw = os.getenv("DATABASE_URL")
    if not raw:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it to the PostgreSQL connection string, e.g. "
            "'postgresql://appuser:dbuser123@localhost:5000/myapp'."
        )

    url = _normalize_database_url(raw)
    return create_engine(url, pool_pre_ping=True)


ENGINE = _build_engine()
SessionLocal = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)


def _parse_pagination(page: int, size: int) -> Tuple[int, int]:
    """Convert 1-based page and size into (limit, offset)."""
    safe_page = max(page, 1)
    safe_size = min(max(size, 1), 100)
    offset = (safe_page - 1) * safe_size
    return safe_size, offset


def _csv_ingredients(raw: str) -> List[str]:
    """Parse a comma-separated list of ingredients, normalized to lower-case."""
    parts = [p.strip() for p in raw.split(",")]
    return [p.lower() for p in parts if p]


# PUBLIC_INTERFACE
def get_db() -> Session:
    """FastAPI dependency providing a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CategoryOut(BaseModel):
    id: int = Field(..., description="Category ID.")
    name: str = Field(..., description="Category name.")


class IngredientOut(BaseModel):
    id: int = Field(..., description="Ingredient ID.")
    name: str = Field(..., description="Ingredient name.")
    quantity: Optional[str] = Field(None, description="Ingredient quantity for the recipe.")


class RecipeListItemOut(BaseModel):
    id: int = Field(..., description="Recipe ID.")
    title: str = Field(..., description="Recipe title.")
    description: Optional[str] = Field(None, description="Short description.")
    category_id: Optional[int] = Field(None, description="Category ID.")
    category_name: Optional[str] = Field(None, description="Category name.")
    avg_rating: Optional[float] = Field(None, description="Average rating score (1-5).")
    ratings_count: int = Field(..., description="Number of ratings for the recipe.")


class RecipeDetailOut(BaseModel):
    id: int = Field(..., description="Recipe ID.")
    title: str = Field(..., description="Recipe title.")
    description: Optional[str] = Field(None, description="Short description.")
    instructions: Optional[str] = Field(None, description="Preparation instructions.")
    category_id: Optional[int] = Field(None, description="Category ID.")
    category_name: Optional[str] = Field(None, description="Category name.")
    ingredients: List[IngredientOut] = Field(default_factory=list, description="Ingredients for the recipe.")
    avg_rating: Optional[float] = Field(None, description="Average rating score (1-5).")
    ratings_count: int = Field(..., description="Number of ratings for the recipe.")


class RatingCreateIn(BaseModel):
    user_name: str = Field(..., min_length=1, max_length=100, description="Display name of the user submitting the rating.")
    score: int = Field(..., ge=1, le=5, description="Rating score from 1 to 5.")
    comment: Optional[str] = Field(None, max_length=2000, description="Optional rating comment.")


class RatingOut(BaseModel):
    id: int = Field(..., description="Rating ID.")
    recipe_id: int = Field(..., description="Recipe ID being rated.")
    user_name: str = Field(..., description="User display name.")
    score: int = Field(..., ge=1, le=5, description="Rating score from 1 to 5.")
    comment: Optional[str] = Field(None, description="Optional comment.")
    created_at: str = Field(..., description="Timestamp when rating was created (ISO 8601 string).")


app = FastAPI(
    title="Recipe Explorer API",
    description=(
        "Backend API for browsing recipes, searching by ingredients, "
        "and submitting/viewing user ratings."
    ),
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
)

# CORS: allow the React frontend (default dev port 3000) and backend served docs.
# If needed, set FRONTEND_ORIGIN to the deployed frontend URL.
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"], summary="Health check")
# PUBLIC_INTERFACE
def health_check() -> Dict[str, str]:
    """Return a simple health check response."""
    return {"message": "Healthy"}


@app.get("/categories", response_model=List[CategoryOut], tags=["Categories"], summary="List categories")
# PUBLIC_INTERFACE
def list_categories(db: Session = Depends(get_db)) -> List[CategoryOut]:
    """Return all recipe categories sorted by name."""
    try:
        rows = db.execute(text("SELECT id, name FROM categories ORDER BY name ASC")).mappings().all()
        return [CategoryOut(**dict(r)) for r in rows]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.get(
    "/recipes",
    response_model=Dict[str, Any],
    tags=["Recipes"],
    summary="List recipes",
    description="List recipes with optional filters and pagination.",
)
# PUBLIC_INTERFACE
def list_recipes(
    category_id: Optional[int] = Query(None, description="Filter recipes by category ID."),
    ingredient: Optional[str] = Query(None, description="Filter recipes containing an ingredient (case-insensitive, partial match)."),
    page: int = Query(1, ge=1, description="1-based page number."),
    size: int = Query(12, ge=1, le=100, description="Page size (max 100)."),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    List recipes with optional filters:
    - category_id
    - ingredient (partial match)
    Pagination:
    - page, size
    """
    limit, offset = _parse_pagination(page, size)

    where_clauses: List[str] = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if category_id is not None:
        where_clauses.append("r.category_id = :category_id")
        params["category_id"] = category_id

    if ingredient:
        where_clauses.append(
            """
            EXISTS (
              SELECT 1
              FROM recipe_ingredients ri
              JOIN ingredients i ON i.id = ri.ingredient_id
              WHERE ri.recipe_id = r.id
                AND lower(i.name) LIKE :ingredient
            )
            """
        )
        params["ingredient"] = f"%{ingredient.strip().lower()}%"

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join([f"({c.strip()})" for c in where_clauses])

    # Count total
    count_sql = f"SELECT COUNT(*) AS total FROM recipes r {where_sql}"
    list_sql = f"""
        SELECT
          r.id,
          r.title,
          r.description,
          r.category_id,
          c.name AS category_name,
          ROUND(AVG(rt.score)::numeric, 2)::float AS avg_rating,
          COUNT(rt.id) AS ratings_count
        FROM recipes r
        LEFT JOIN categories c ON c.id = r.category_id
        LEFT JOIN ratings rt ON rt.recipe_id = r.id
        {where_sql}
        GROUP BY r.id, c.name
        ORDER BY r.id DESC
        LIMIT :limit OFFSET :offset
    """

    try:
        total = int(db.execute(text(count_sql), params).scalar_one())
        rows = db.execute(text(list_sql), params).mappings().all()
        items = [RecipeListItemOut(**dict(r)) for r in rows]
        return {"page": page, "size": size, "total": total, "items": items}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.get(
    "/recipes/{id}",
    response_model=RecipeDetailOut,
    tags=["Recipes"],
    summary="Get recipe detail",
)
# PUBLIC_INTERFACE
def get_recipe(id: int, db: Session = Depends(get_db)) -> RecipeDetailOut:
    """Return a single recipe with ingredients and rating summary."""
    recipe_sql = """
        SELECT
          r.id,
          r.title,
          r.description,
          r.instructions,
          r.category_id,
          c.name AS category_name,
          ROUND(AVG(rt.score)::numeric, 2)::float AS avg_rating,
          COUNT(rt.id) AS ratings_count
        FROM recipes r
        LEFT JOIN categories c ON c.id = r.category_id
        LEFT JOIN ratings rt ON rt.recipe_id = r.id
        WHERE r.id = :recipe_id
        GROUP BY r.id, c.name
    """
    ingredients_sql = """
        SELECT
          i.id,
          i.name,
          ri.quantity
        FROM recipe_ingredients ri
        JOIN ingredients i ON i.id = ri.ingredient_id
        WHERE ri.recipe_id = :recipe_id
        ORDER BY i.name ASC
    """

    try:
        recipe_row = db.execute(text(recipe_sql), {"recipe_id": id}).mappings().first()
        if not recipe_row:
            raise HTTPException(status_code=404, detail="Recipe not found")

        ing_rows = db.execute(text(ingredients_sql), {"recipe_id": id}).mappings().all()
        recipe = dict(recipe_row)
        recipe["ingredients"] = [IngredientOut(**dict(r)) for r in ing_rows]
        return RecipeDetailOut(**recipe)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.get(
    "/search",
    response_model=List[RecipeListItemOut],
    tags=["Search"],
    summary="Search recipes by ingredients",
    description=(
        "Search recipes matching ALL provided ingredients. "
        "Pass ingredients as a comma-separated list."
    ),
)
# PUBLIC_INTERFACE
def search_recipes(
    ingredients: str = Query(..., description="Comma-separated list of ingredient names. Matches ALL ingredients."),
    page: int = Query(1, ge=1, description="1-based page number."),
    size: int = Query(12, ge=1, le=100, description="Page size (max 100)."),
    db: Session = Depends(get_db),
) -> List[RecipeListItemOut]:
    """Return recipes that contain all specified ingredients (case-insensitive exact match on ingredient names)."""
    requested = _csv_ingredients(ingredients)
    if not requested:
        return []

    limit, offset = _parse_pagination(page, size)

    # We match ALL ingredients by requiring the count of distinct matched ingredients equals len(requested).
    # Ingredient match is case-insensitive exact name match (normalized to lower()).
    sql = """
        WITH matched AS (
          SELECT
            r.id AS recipe_id
          FROM recipes r
          JOIN recipe_ingredients ri ON ri.recipe_id = r.id
          JOIN ingredients i ON i.id = ri.ingredient_id
          WHERE lower(i.name) = ANY(:ingredient_list)
          GROUP BY r.id
          HAVING COUNT(DISTINCT lower(i.name)) = :ingredient_count
        )
        SELECT
          r.id,
          r.title,
          r.description,
          r.category_id,
          c.name AS category_name,
          ROUND(AVG(rt.score)::numeric, 2)::float AS avg_rating,
          COUNT(rt.id) AS ratings_count
        FROM matched m
        JOIN recipes r ON r.id = m.recipe_id
        LEFT JOIN categories c ON c.id = r.category_id
        LEFT JOIN ratings rt ON rt.recipe_id = r.id
        GROUP BY r.id, c.name
        ORDER BY r.id DESC
        LIMIT :limit OFFSET :offset
    """

    try:
        rows = (
            db.execute(
                text(sql),
                {
                    "ingredient_list": requested,
                    "ingredient_count": len(requested),
                    "limit": limit,
                    "offset": offset,
                },
            )
            .mappings()
            .all()
        )
        return [RecipeListItemOut(**dict(r)) for r in rows]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.post(
    "/recipes/{id}/ratings",
    response_model=RatingOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Ratings"],
    summary="Add a rating to a recipe",
)
# PUBLIC_INTERFACE
def create_rating(id: int, payload: RatingCreateIn, db: Session = Depends(get_db)) -> RatingOut:
    """Create a new rating for the specified recipe."""
    try:
        exists = db.execute(text("SELECT 1 FROM recipes WHERE id = :id"), {"id": id}).first()
        if not exists:
            raise HTTPException(status_code=404, detail="Recipe not found")

        insert_sql = """
            INSERT INTO ratings (recipe_id, user_name, score, comment)
            VALUES (:recipe_id, :user_name, :score, :comment)
            RETURNING id, recipe_id, user_name, score, comment, created_at
        """
        row = (
            db.execute(
                text(insert_sql),
                {
                    "recipe_id": id,
                    "user_name": payload.user_name,
                    "score": payload.score,
                    "comment": payload.comment,
                },
            )
            .mappings()
            .one()
        )
        db.commit()

        # created_at comes from PostgreSQL as datetime; serialize to ISO string.
        created_at = row["created_at"].isoformat()
        out = dict(row)
        out["created_at"] = created_at
        return RatingOut(**out)
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e


@app.get(
    "/recipes/{id}/ratings",
    response_model=List[RatingOut],
    tags=["Ratings"],
    summary="List ratings for a recipe",
)
# PUBLIC_INTERFACE
def list_ratings(id: int, db: Session = Depends(get_db)) -> List[RatingOut]:
    """Return all ratings for a recipe (newest first)."""
    try:
        exists = db.execute(text("SELECT 1 FROM recipes WHERE id = :id"), {"id": id}).first()
        if not exists:
            raise HTTPException(status_code=404, detail="Recipe not found")

        sql = """
            SELECT id, recipe_id, user_name, score, comment, created_at
            FROM ratings
            WHERE recipe_id = :recipe_id
            ORDER BY created_at DESC, id DESC
        """
        rows = db.execute(text(sql), {"recipe_id": id}).mappings().all()
        out: List[RatingOut] = []
        for r in rows:
            d = dict(r)
            d["created_at"] = d["created_at"].isoformat()
            out.append(RatingOut(**d))
        return out
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}") from e
