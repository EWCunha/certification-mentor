def hit_rate(relevance: list[list[int]]) -> float:
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)


def mrr(relevance: list[list[int]]) -> float:
    """
    Mean Reciprocal Rank
    """

    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)
