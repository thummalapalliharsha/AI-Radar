from backend.app.services.story_clustering import cluster_articles
from backend.app.utils.database import get_connection

connection = get_connection()

rows = connection.execute(
    "SELECT * FROM articles "
    "WHERE is_ai_news=1 AND should_show=1 "
    "ORDER BY published_timestamp DESC LIMIT 100"
).fetchall()

connection.close()

articles = [dict(row) for row in rows]

clusters = cluster_articles(articles)

print()
print("=" * 70)
print("MULTI-ARTICLE STORY CLUSTERS")
print("=" * 70)

for i, cluster in enumerate(clusters, 1):
    member_count = cluster.get("member_count", 1)

    if member_count > 1:
        headline = cluster.get(
            "canonical_headline",
            cluster.get("title", "Untitled")
        )

        sources = cluster.get("sources", [])

        print()
        print(f"CLUSTER {i}")
        print(f"Headline : {headline}")
        print(f"Members  : {member_count}")
        print(f"Sources  : {len(sources)}")
        print(f"Source names: {sources}")

print()
print("=" * 70)
print(f"Articles considered : {len(articles)}")
print(f"Story clusters      : {len(clusters)}")
print("=" * 70)



