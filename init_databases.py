#!/usr/bin/env python3
"""
Скрипт инициализации баз данных в правильном месте.
"""

import os
import sys

# Добавляем корневой каталог в путь для импортов
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database.init_db import create_database_manager
from src.database.vector_store import initialize_vector_store
from src.database.graph_store import initialize_graph_store
from src.config import settings

def main():
    """Инициализирует все базы данных."""
    print("Инициализация баз данных ATS...")

    # 1. SQLite база данных
    print(f"\nSQLite: {settings.sync_db_path}")
    db_manager = create_database_manager(settings.sync_db_path)

    if db_manager.initialize_database():
        print("OK: SQLite база данных инициализирована успешно")

        # Проверяем здоровье
        health = db_manager.check_database_health()
        print(f"   Статус: {health['status']}")
        print(f"   Таблицы: {len(health['tables'])}")
        print(f"   Записей: {sum(health['record_counts'].values())}")
    else:
        print("ERROR: Ошибка инициализации SQLite БД")
        return False

    # 2. ChromaDB векторная база данных
    print(f"\nChromaDB: {settings.chromadb_persist_dir}")
    try:
        initialize_vector_store(settings.chromadb_persist_dir)

        from src.database.vector_store import vector_store
        if vector_store and vector_store.initialize_collections():
            print("OK: ChromaDB инициализирована успешно")

            # Проверяем статистику
            stats = vector_store.get_collection_stats()
            print(f"   Коллекции: {len(stats)}")
            print(f"   Общее количество записей: {sum(s['count'] for s in stats.values())}")
        else:
            print("ERROR: Ошибка инициализации ChromaDB")
            return False

    except Exception as e:
        print(f"ERROR: Ошибка ChromaDB: {e}")
        print("   Это нормально - ChromaDB может потребовать дополнительной настройки")

    # 3. Memgraph графовая база данных
    print(f"\nMemgraph: {settings.memgraph_uri}")
    try:
        initialize_graph_store(
            settings.memgraph_uri,
            settings.memgraph_user,
            settings.memgraph_password,
        )

        from src.database.graph_store import graph_store
        if graph_store:
            print("OK: Memgraph инициализирована успешно")
        else:
            print("WARNING: Memgraph не доступна (требуется Docker контейнер)")

    except Exception as e:
        print(f"WARNING: Memgraph не доступна: {e}")
        print("   Запустите Docker контейнер: docker run -p 7687:7687 memgraph/memgraph:latest")

    print("\nИнициализация завершена!")
    print("\nСтруктура баз данных:")
    print(f"   SQLite: {settings.sync_db_path}")
    print(f"   ChromaDB: {settings.chromadb_persist_dir}")
    print(f"   Memgraph: {settings.memgraph_uri}")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
