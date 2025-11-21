from pathlib import Path

from duckdb import DuckDBPyConnection
from loguru import logger

from src.utils.cache import save_tensor, load_tensor


class UserLoader:
    def __init__(self, con: DuckDBPyConnection, out_dir: Path):
        self.con = con
        self.out_dir = out_dir

    def load(self):
        """
        Loads ordered users df and caches user_ids.pt for fast reload.

        Returns:
            df:       DataFrame of users ORDERED BY id
            user_ids: list of user ids in correct order
        """

        cache_path = self.out_dir / "user_ids.pt"

        df = self.con.execute("SELECT * FROM users ORDER BY id").df()

        if cache_path.exists():
            logger.info("Loading cached user_ids.pt...")
            user_ids = load_tensor(cache_path, not_tensor=True)

            db_ids = df["id"].tolist()
            if user_ids != db_ids:
                logger.warning(
                    "Cached user_ids.pt does NOT match database ordering — rebuilding."
                )
                user_ids = db_ids
                save_tensor(user_ids, cache_path, not_tensor=True)
            else:
                logger.success("User IDs validated against database.")
        else:
            # First-time creation
            logger.info("Saving user_ids.pt...")
            user_ids = df["id"].tolist()
            save_tensor(user_ids, cache_path, not_tensor=True)
            logger.success("user_ids.pt created.")

        return df, user_ids
