def get_rolled_and_unrolled_data(input_data, args):
    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    all_unrolled = []
    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}


    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []
        dep_tags = []
        dep_index = []
        dep_dirs = []

        # Classify based on POS-tags

        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            # We would tokenize the aspect while at it.
            aspect = word_tokenize(aspect)
            sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            # Center on the aspect.
            dep_tag, dep_idx, dep_dir = reshape_dependency_tree(frm, to, e['dependencies'], tokens=e['tokens'], max_hop=args.max_hop)

            # Because of tokenizer differences, aspect opsitions are off, so we find the index and try again.
            if len(dep_tag) == 0:
                zero_dep_counter += 1
                as_sent = e['aspect_sentiment'][i][0].split()
                as_start = e['tokens'].index(as_sent[0])
                # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
                as_end = e['tokens'].index(
                    as_sent[-1]) if len(as_sent) > 1 else as_start + 1
                print("Debugging: as_start as_end ", as_start, as_end)
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree(as_start, as_end, e['dependencies'], tokens=e['tokens'], max_hop=args.max_hop)
                if len(dep_tag) == 0:  # for debugging
                    print("Debugging: zero_dep",
                          e['aspect_sentiment'][i][0], e['tokens'])
                    print("Debugging: ". e['dependencies'])
                else:
                    zero_dep_counter -= 1

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies']})

    return all_unrolled