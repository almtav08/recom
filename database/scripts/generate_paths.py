import random


# Base student paths
mandatory = [2, 5, 7, 8, 11, 15, 17, 18, 21, 24]  # Interacts with only the mandatory items
complete = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]  # Interacts with all items
FINAL_ITEM = 29

# region Pass path transformations
# These transformations are used to create paths that are still considered "Pass" paths
# 1. Partial shuffle: Randomly shuffles segments of the path
# 2. Controlled extension: Randomly inserts nodes from available_items at random positions


def partial_shuffle(path, window=2):
    """
    Shuffles segments of a path while maintaining the overall structure.

    Args:
        path: List of integers representing the original path
        window: Size of each segment to shuffle (default: 2)
              Larger values create more randomization

    Returns:
        A new path with shuffled segments
    """
    # Create a copy of the original path to avoid modifying it
    new_path = path[:]

    # Creates a random skip value between 0 and window - 1
    window_skip = random.randint(0, window)

    # Iterate through the path in steps of 'window' size
    for i in range(0, len(path) - window + 1, window + window_skip):
        # Extract the current segment
        segment = new_path[i : i + window]

        # Randomly reorder the elements in the segment
        random.shuffle(segment)

        # Replace the original segment with the shuffled one
        new_path[i : i + window] = segment

    return new_path


def controlled_extension(path, available_items, max_additional_items=3):
    """
    Extends a path by randomly inserting nodes from available_items at random positions,
    ensuring that no consecutive elements are the same.

    Args:
        path: List of integers representing the original path
        available_items: List of all available items that can be inserted
        max_additional_items: Maximum number of items to insert (default: 3)
    Returns:
        A new extended path with random elements inserted at random positions
    """
    if not path or not available_items:
        return path[:]

    # Create a copy of the original path
    new_path = path[:]

    # Decide how many items to insert (between 0 and max_additional_items)
    num_to_insert = random.randint(0, max_additional_items)

    # Perform the insertions one by one
    for _ in range(num_to_insert):
        # Choose a random position to insert (between 0 and current length)
        # This allows inserting at the beginning, middle, or end
        current_length = len(new_path)
        insert_position = random.randint(0, current_length)

        # Determine which items would cause consecutive repeats
        invalid_items = set()

        # Check element before the insertion point (if any)
        if insert_position > 0:
            invalid_items.add(new_path[insert_position - 1])

        # Check element after the insertion point (if any)
        if insert_position < current_length:
            invalid_items.add(new_path[insert_position])

            # Add restriction: element to insert cannot be greater than the current element at insert_position
            current_element = new_path[insert_position]
            invalid_items.update(
                [item for item in available_items if item > current_element]
            )

        # Filter available items to avoid consecutive repetition and value restriction
        valid_choices = [item for item in available_items if item not in invalid_items]

        # If no valid choices are available, skip this insertion
        if not valid_choices:
            continue

        # Select a random item from valid choices
        item_to_insert = random.choice(valid_choices)

        # Insert the item at the chosen position
        new_path.insert(insert_position, item_to_insert)

    return new_path


# endregion

# region Redemption path transformations
# These transformations are used to create paths that are considered "Pass" paths but have started as "Fail"


def redemption_path(
    critical_elements,
    all_elements,
    max_random_length=7,
    min_random_portion=0.3,
    max_random_portion=0.5,
):
    """
    Creates a redemption path combining a random first portion with a strategic second portion
    that ensures all critical elements are visited.

    Args:
        critical_elements: List of critical elements that must be visited for a path to succeed
        all_elements: List of all available elements that can be included in the path
        max_random_length: Maximum length of the random portion (default: 7)
        min_random_portion: Minimum portion of the path that should be random (default: 0.3)
        max_random_portion: Maximum portion of the path that should be random (default: 0.5)

    Returns:
        A new path with a random first portion and a strategic second portion
    """
    # Create an empty path
    new_path = []

    # Decide how much of the path should be random (between min and max portion)
    random_portion = random.uniform(min_random_portion, max_random_portion)
    random_length = max(1, int(len(max_random_length) * random_portion))

    # Create the random first portion of the path
    # This can include any elements, potentially repeating elements from current_path
    available_for_random = all_elements[:]

    # Make sure we don't use too many elements in the random portion
    # so there are still enough elements left for the strategic portion
    if random_length > len(available_for_random) // 2:
        random_length = len(available_for_random) // 2

    # Generate the random first portion
    random_part = []
    print("Random length:", random_length)
    for _ in range(random_length):
        if available_for_random:
            random_element = random.choice(available_for_random)
            random_part.append(random_element)

    # Determine which critical elements have already been seen in the current path
    seen_critical = set(random_part).intersection(set(critical_elements))

    # Find critical elements that haven't been seen yet
    missing_critical = [elem for elem in critical_elements if elem not in seen_critical]

    # Find other elements that haven't been seen yet
    seen_elements = set(random_part)
    missing_other = [
        elem
        for elem in all_elements
        if elem not in seen_elements and elem not in missing_critical
    ]

    # Now create the strategic second portion to ensure all missing critical elements are included
    strategic_part = missing_critical[:]

    # Optionally add some non-critical missing elements to make the path more interesting
    # but not too many to keep the focus on critical elements
    num_other_to_add = min(len(missing_other), len(missing_critical) // 2)
    if num_other_to_add > 0 and missing_other:
        strategic_part.extend(random.sample(missing_other, num_other_to_add))

    # Shuffle the strategic part to make it less predictable
    # while still ensuring all critical elements are included
    partial_shuffle(strategic_part)

    # Combine the parts to form the redemption path
    new_path = random_part + strategic_part

    return new_path


# endregion

# region Fail path transformations
# These transformations are used to create paths that are considered "Fail" paths
# 1. Controlled removal: Randomly removes key elements from the path
# 2. Random swaps: Randomly swaps elements in the path a random number of times
# 3. Full randomization: Completely randomizes the order of elements in the path


def controlled_removal(path, key_elements, max_items_to_remove=5):
    """
    Removes key elements from a path.

    Args:
        path: List of integers representing the original path
        key_elements: List of specific elements that should be removed
        max_items_to_remove: Maximum number of key elements to remove (default: 5)

    Returns:
        A new path with some key elements randomly removed
    """
    if not path or max_items_to_remove <= 0:
        return path[:]

    # Create a copy of the original path
    new_path = path[:]

    # Find indices of key elements that can be removed
    # Only elements that are in the key_elements list are considered
    removable_indices = [i for i, item in enumerate(new_path) if item in key_elements]

    # If no key elements are found in the path, return the original path
    if not removable_indices:
        return new_path

    # Decide how many key elements to remove (between 0 and max_items_to_remove)
    # Cannot remove more than the available key elements
    num_to_remove = random.randint(0, min(max_items_to_remove, len(removable_indices)))

    if num_to_remove == 0:
        return new_path

    # Randomly select indices of key elements to remove
    indices_to_remove = random.sample(removable_indices, num_to_remove)

    # Sort indices in descending order to avoid index shifting when removing items
    indices_to_remove.sort(reverse=True)

    # Remove only the selected key elements
    for index in indices_to_remove:
        new_path.pop(index)

    return new_path


def random_swaps(path, max_swaps=5):
    """
    Randomly swaps elements in a path a random number of times.

    This function performs between 0 and max_swaps random swaps,
    where each swap exchanges two elements at random positions in the path.

    Args:
        path: List of integers representing the original path
        max_swaps: Maximum number of swaps to perform (default: 3)

    Returns:
        A new path with randomly swapped elements
    """
    if not path or len(path) < 2 or max_swaps <= 0:
        return path[:]

    # Create a copy of the original path
    new_path = path[:]

    # Decide how many swaps to perform (between 1 and max_swaps)
    num_swaps = random.randint(1, max_swaps)

    # Perform the swaps
    for _ in range(num_swaps):
        # Choose two random different positions to swap
        # random.sample ensures the positions are different
        pos1, pos2 = random.sample(range(len(new_path)), 2)

        # Swap the elements
        new_path[pos1], new_path[pos2] = new_path[pos2], new_path[pos1]

    return new_path


def full_randomization(path):
    """
    Completely randomizes the order of elements in a path.

    Args:
        path: List of integers representing the original path

    Returns:
        A new path with elements in completely random order
    """
    if not path or len(path) < 2:
        return path[:]

    # Create a copy of the original path
    new_path = path[:]

    # Completely shuffle the path
    random.shuffle(new_path)

    return new_path


# endregion


print("Pass transformations:")
print(partial_shuffle(mandatory))
print(controlled_extension(mandatory, complete))

print("\nRedemption transformations:")
print(redemption_path(mandatory, mandatory, complete))

print("\nFail transformations:")
print(controlled_removal(mandatory, mandatory))
print(random_swaps(mandatory))
print(full_randomization(mandatory))
