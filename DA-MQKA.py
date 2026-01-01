from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import hashlib

# ============================================================================
# Common Constants and Mappings
# ============================================================================

# Bell state mapping (used in four-particle protocol)
bell_map = {
    '00': '00',  # B00
    '01': '10',  # B10
    '10': '01',  # B01
    '11': '11'  # B11
}

# Bell measurement results to Bell state mapping (used in three-particle protocol)
BELL_MEASUREMENT_TO_STATE = {
    '00': 'B00',
    '01': 'B10',
    '10': 'B01',
    '11': 'B11'
}

# GHZ-like state to Bell state mapping based on Z measurement (three-particle)
GHZ_LIKE_TO_BELL_STATE = {
    0: {'0': 'B00', '1': 'B10'},
    1: {'0': 'B10', '1': 'B00'},
    2: {'0': 'B00', '1': 'B10'},
    3: {'0': 'B10', '1': 'B00'},
    4: {'0': 'B01', '1': 'B11'},
    5: {'0': 'B11', '1': 'B01'},
    6: {'0': 'B01', '1': 'B11'},
    7: {'0': 'B11', '1': 'B01'},
}

# Single photon states and their measurement bases
SINGLE_PHOTON_STATES = {
    '0': {'basis': 'Z', 'state': '0'},
    '1': {'basis': 'Z', 'state': '1'},
    '+': {'basis': 'X', 'state': '0'},
    '-': {'basis': 'X', 'state': '1'}
}

backend = Aer.get_backend("qasm_simulator")


# ============================================================================
# Common Utility Functions
# ============================================================================

def flip_first_bit(key):
    """Flip the first bit of a two-bit key"""
    return ('1' if key[0] == '0' else '0') + key[1]


def flip_second_bit(key):
    """Flip the second bit of a two-bit key"""
    return key[0] + ('1' if key[1] == '0' else '0')


def flip_both_bits(key):
    """Flip both bits of a two-bit key"""
    return ''.join('1' if bit == '0' else '0' for bit in key)


def find_different_positions(key1, key2):
    """Find positions where two keys differ"""
    return [i for i, (b1, b2) in enumerate(zip(key1, key2)) if b1 != b2]


def flip_bits(key, diff_positions):
    """Flip bits at specified positions"""
    key_list = list(key)
    for pos in diff_positions:
        if pos < len(key_list):
            key_list[pos] = '1' if key_list[pos] == '0' else '0'
    return ''.join(key_list)


def format_table(headers, rows):
    """Format data into a table with borders"""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    format_str = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    table = []
    table.append(separator)
    table.append(format_str.format(*headers))
    table.append(separator)
    for row in rows:
        table.append(format_str.format(*row))
    table.append(separator)

    return "\n".join(table)


# ============================================================================
# Hash Functions (Common)
# ============================================================================

def h_shake256(data, output_length):
    """
    Generate hash values of specified length using SHAKE256 XOF function

    Args:
        data: Input data (string or bytes)
        output_length: Output length (bits)

    Returns:
        Binary string, length strictly equal to output_length
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    shake = hashlib.shake_256()
    shake.update(data)
    bytes_needed = (output_length + 7) // 8
    hash_bytes = shake.digest(bytes_needed)
    hash_bits = ''.join(format(byte, '08b') for byte in hash_bytes)
    return hash_bits[:output_length]


def hash_commitment(data):
    """
    Generate hash commitment

    Args:
        data: Input data (string)

    Returns:
        Hash value (hexadecimal string)
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


# ============================================================================
# Three-Particle GHZ-like State Functions
# ============================================================================

def create_ghz_like_state_three(type_num):
    """Create a three-particle GHZ-like state"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    qc.h(1)
    qc.h(2)

    if type_num == 0:
        pass
    elif type_num == 1:
        qc.x(0)
    elif type_num == 2:
        qc.z(1)
        qc.z(2)
    elif type_num == 3:
        qc.z(1)
        qc.z(2)
        qc.x(0)
    elif type_num == 4:
        qc.z(0)
        qc.z(2)
    elif type_num == 5:
        qc.z(0)
        qc.z(2)
        qc.x(0)
    elif type_num == 6:
        qc.z(0)
        qc.z(1)
    elif type_num == 7:
        qc.z(0)
        qc.z(1)
        qc.x(0)

    return qc


def z_measure(qc):
    """Perform Z measurement on qubit 0"""
    qc.barrier()
    qc.measure(0, 0)
    return qc


def bell_measure(qc):
    """Perform Bell measurement on qubits 1 and 2"""
    qc.barrier()
    qc.cx(1, 2)
    qc.h(1)
    qc.measure([1, 2], [1, 2])
    return qc


def create_single_photon_state():
    """Create a single photon decoy state"""
    state = random.choice(list(SINGLE_PHOTON_STATES.keys()))
    qc = QuantumCircuit(1, 1)
    if state == '0':
        pass
    elif state == '1':
        qc.x(0)
    elif state == '+':
        qc.h(0)
    elif state == '-':
        qc.x(0)
        qc.h(0)
    return qc, state


def insert_single_photons(sequence, num_insertions):
    """Insert decoy states (single photons) into the sequence"""
    if num_insertions <= len(sequence):
        positions = random.sample(range(len(sequence)), num_insertions)
    else:
        positions = random.choices(range(len(sequence)), k=num_insertions)

    positions.sort()
    new_sequence = []
    decoy_info = []
    current_pos = 0

    for pos in positions:
        new_sequence.extend(sequence[current_pos:pos])
        decoy, state = create_single_photon_state()
        new_sequence.append(decoy)
        decoy_info.append((len(new_sequence) - 1, state))
        current_pos = pos

    new_sequence.extend(sequence[current_pos:])
    return new_sequence, decoy_info


def measure_single_photon(qc, basis):
    """Measure a single photon in the given basis"""
    if basis == 'X':
        qc.h(0)
    qc.measure(0, 0)
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=1024).result()
    counts = result.get_counts()
    return max(counts, key=counts.get)[::-1][0]


def verify_single_photons(sequence, decoy_info):
    """Verify decoy states by measuring them"""
    error = 0
    total_checks = 0

    for pos, state in decoy_info:
        basis = SINGLE_PHOTON_STATES[state]['basis']
        expected = SINGLE_PHOTON_STATES[state]['state']
        result = measure_single_photon(sequence[pos], basis)

        if result != expected:
            error += 1
            print(f"Error at position {pos}: Expected {expected}, Got {result}")
        total_checks += 1

    error_rate = error / total_checks
    print(f"Decoy state error rate: {error_rate:.2%} ({error}/{total_checks} errors)")
    return error_rate < 0.15


def remove_single_photons(sequence, decoy_info):
    """Remove decoy states from the sequence"""
    decoy_positions = [pos for pos, _ in decoy_info]
    return [qc for i, qc in enumerate(sequence) if i not in decoy_positions]


def simulate_pair_three(M, sender, receiver):
    """Simulate key generation for a pair using three-particle GHZ-like states"""
    k_sender, k_receiver = [], []
    used_ghz_types = set()

    for _ in range(M):
        ghz_type = random.randint(0, 7)
        used_ghz_types.add(ghz_type)
        qc = create_ghz_like_state_three(ghz_type)
        z_measure(qc)
        bell_measure(qc)

        compiled = transpile(qc, backend)
        result = backend.run(compiled, shots=1024).result()
        counts = result.get_counts()
        measurement = max(counts, key=counts.get)[::-1]

        res_z = measurement[0]
        res_bell = measurement[1] + measurement[2]

        bell_sender = GHZ_LIKE_TO_BELL_STATE[ghz_type][res_z]
        bell_receiver = BELL_MEASUREMENT_TO_STATE[res_bell]

        k_sender.append(bell_sender[1:])
        k_receiver.append(bell_receiver[1:])

    return ''.join(k_sender), ''.join(k_receiver), used_ghz_types


# ============================================================================
# Four-Particle GHZ-like State Functions
# ============================================================================

def prepare_ghz_like_four(qc, qubits, state_type):
    """Prepare a four-particle GHZ-like state"""
    if state_type in [0, 1, 2, 3]:
        # prepare GHZ-like0 = (|++++> + |---->)/√2
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[2])
        qc.cx(qubits[0], qubits[3])
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.h(qubits[2])
        qc.h(qubits[3])

        if state_type == 1:
            qc.x(qubits[0])  # GHZ-like1
        elif state_type == 2:
            qc.z(qubits[3])  # GHZ-like2
        elif state_type == 3:
            qc.z(qubits[3])
            qc.x(qubits[0])  # GHZ-like3

    elif state_type in [4, 5, 6, 7]:
        # prepare GHZ-like4 = (|++--> + |--++>)/√2
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[2])
        qc.cx(qubits[0], qubits[3])
        qc.z(qubits[2])
        qc.z(qubits[3])
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.h(qubits[2])
        qc.h(qubits[3])

        if state_type == 5:
            qc.x(qubits[0])  # GHZ-like5
        elif state_type == 6:
            qc.z(qubits[3])  # GHZ-like6
        elif state_type == 7:
            qc.z(qubits[3])
            qc.x(qubits[0])  # GHZ-like7

    elif state_type in [8, 9, 10, 11]:
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[2])
        qc.cx(qubits[0], qubits[3])
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.h(qubits[2])
        qc.h(qubits[3])

        if state_type == 8:
            qc.x(qubits[1])
            qc.x(qubits[3])
            qc.z(qubits[1])
            qc.z(qubits[3])  # GHZ-like8 = (|+-+-> + |-+-+>)/√2
        elif state_type == 9:
            qc.x(qubits[1])
            qc.x(qubits[3])
            qc.z(qubits[1])
            qc.z(qubits[3])
            qc.x(qubits[0])  # GHZ-like9
        elif state_type == 10:
            qc.z(qubits[1])  # GHZ-like10
        elif state_type == 11:
            qc.z(qubits[1])
            qc.x(qubits[0])  # GHZ-like11

    elif state_type in [12, 13, 14, 15]:
        # GHZ-like12 = (|+--+> + |-++->)/√2
        qc.h(qubits[0])
        qc.cx(qubits[0], qubits[1])
        qc.cx(qubits[0], qubits[2])
        qc.cx(qubits[0], qubits[3])
        qc.h(qubits[0])
        qc.h(qubits[1])
        qc.h(qubits[2])
        qc.h(qubits[3])
        qc.z(qubits[1])
        qc.z(qubits[2])

        if state_type == 13:
            qc.x(qubits[0])  # GHZ-like13
        elif state_type == 14:
            qc.z(qubits[3])  # GHZ-like14
        elif state_type == 15:
            qc.z(qubits[3])
            qc.x(qubits[0])  # GHZ-like15


def prepare_decoy_state(qc, qubit, state_type):
    """Prepare a decoy state |D> on the given qubit"""
    if state_type == 0:  # |0>
        pass
    elif state_type == 1:  # |1>
        qc.x(qubit)
    elif state_type == 2:  # |+>
        qc.h(qubit)
    elif state_type == 3:  # |->
        qc.x(qubit)
        qc.h(qubit)


def measure_decoy_state(qc, qubit, creg, basis):
    """Measure a decoy state in the given basis"""
    if basis == 1:  # Z basis
        qc.measure(qubit, creg)
    else:  # X basis
        qc.h(qubit)
        qc.measure(qubit, creg)


def calculate_error_rate(original_states, measurement_results, bases):
    """Calculate the error rate of decoy state measurements"""
    errors = 0
    total = len(original_states)

    for i in range(total):
        if bases[i] == 0:  # Z basis
            if original_states[i] in [0, 1] and measurement_results[i] != original_states[i]:
                errors += 1
        else:  # X basis
            if original_states[i] == 2 and measurement_results[i] == 1:
                errors += 1
            elif original_states[i] == 3 and measurement_results[i] == 0:
                errors += 1

    return errors, total, (errors / total) if total else 0


def distribute_key_between_participants_four(sender, receiver, M, d):
    """Distribute key between two participants using four-particle GHZ-like states and decoy states"""
    backend = Aer.get_backend('qasm_simulator')

    # First handle decoy states
    decoy_states = [random.randint(0, 3) for _ in range(d)]
    decoy_bases = [random.randint(0, 1) for _ in range(d)]

    # Measure decoy states
    measurement_results = []
    for state, basis in zip(decoy_states, decoy_bases):
        qreg = QuantumRegister(1)
        creg = ClassicalRegister(1)
        qc = QuantumCircuit(qreg, creg)

        prepare_decoy_state(qc, qreg[0], state)
        measure_decoy_state(qc, qreg[0], creg[0], basis)

        compiled = transpile(qc, backend)
        result = backend.run(compiled, shots=1).result()
        measurement_results.append(int(list(result.get_counts().keys())[0]))

    # Calculate error rate
    errors, total, error_rate = calculate_error_rate(decoy_states, measurement_results, decoy_bases)

    if error_rate > 0.15:
        return None, None, None, None, None, None

    # Now handle GHZ-like states and key generation
    key = ''
    ghz_states = []
    used_ghz_types = set()

    for _ in range(M):
        qreg = QuantumRegister(4)
        creg = ClassicalRegister(4)
        qc = QuantumCircuit(qreg, creg)

        # Prepare GHZ-like state
        state_type = random.randint(0, 15)
        used_ghz_types.add(state_type)
        prepare_ghz_like_four(qc, qreg, state_type)
        ghz_states.append(state_type)

        # Bell measurement for particles 1,2
        qc.cx(qreg[0], qreg[1])
        qc.h(qreg[0])
        qc.measure(qreg[0], creg[0])
        qc.measure(qreg[1], creg[1])

        # Bell measurement for particles 3,4
        qc.cx(qreg[2], qreg[3])
        qc.h(qreg[2])
        qc.measure(qreg[2], creg[2])
        qc.measure(qreg[3], creg[3])

        compiled = transpile(qc, backend)
        result = backend.run(compiled, shots=1).result()
        bits = list(result.get_counts().keys())[0][::-1]  # Qiskit results need to be reversed

        # Get measurement results
        p1_bits = bits[0:2]  # Particles 1 and 2
        p2_bits = bits[2:4]  # Particles 3 and 4

        # Map to key bits
        key_bits = bell_map[p1_bits]

        # Adjust key based on GHZ-like state type
        if state_type in [1, 5, 9, 13]:
            key_bits = flip_first_bit(key_bits)
        elif state_type in [2, 6, 10, 14]:
            key_bits = flip_second_bit(key_bits)
        elif state_type in [3, 7, 11, 15]:
            key_bits = flip_both_bits(key_bits)

        key += key_bits

    return key, error_rate, ghz_states, errors, total, used_ghz_types


# ============================================================================
# Dynamic Adaptive Key Update Mechanism (Common)
# ============================================================================

def scene1_key_update(remaining_parties, KP_old):
    """
    Scene 1: Key Update Mechanism when only members withdraw

    Args:
        remaining_parties: Remaining participants number
        KP_old: Old shared key

    Returns:
        KP_new: New shared key
    """
    print("\n=== Scene 1: Key Update (Participants Withdrawal) ===")
    print(f"Remaining participants: {remaining_parties}")
    print(f"Old key (KP_Old): {KP_old}")

    print("\nStep 1: Generating random numbers and commitments...")
    random_numbers = []
    local_random_numbers = []
    commitments = []

    for i in range(remaining_parties):
        R_x = ''.join(random.choice('01') for _ in range(32))
        R_L = ''.join(random.choice('01') for _ in range(32))
        random_numbers.append(R_x)
        local_random_numbers.append(R_L)

        commitment_data = R_x + R_L
        C_i = hash_commitment(commitment_data)
        commitments.append(C_i)
        print(f"  P{i + 1}: Generated commitment C_{i + 1} = {C_i[:16]}...")

    print("\nStep 2: Revealing random numbers and verifying commitments...")
    valid_reveals = []

    for i in range(remaining_parties):
        R_x = random_numbers[i]
        R_L = local_random_numbers[i]
        commitment_data = R_x + R_L
        C_i_verify = hash_commitment(commitment_data)

        if C_i_verify == commitments[i]:
            valid_reveals.append((i, R_x, R_L))
            print(f"  P{i + 1}: Commitment verified successfully")
        else:
            print(f"  P{i + 1}: Commitment verification failed!")
            return None

    print("\nStep 3: Building valid reveal set...")
    V = valid_reveals
    print(f"  Valid reveals: {len(V)}/{remaining_parties}")

    print("\nStep 4: Concatenating random numbers...")
    sorted_reveals = sorted(V, key=lambda x: x[0])
    R_concat = ''.join(R_x for _, R_x, _ in sorted_reveals)
    print(f"  R_Concat length: {len(R_concat)} bits")

    print("\nStep 5: Computing new key using SHAKE256...")
    key_length = len(KP_old)
    input_data = R_concat + KP_old
    KP_new = h_shake256(input_data, key_length)
    print(f"  New key (KP_New) length: {len(KP_new)} bits")

    print("\nStep 6: Verifying new key consistency...")
    hash_values = []
    for i in range(remaining_parties):
        H_new = hash_commitment(KP_new)
        hash_values.append(H_new)
        print(f"  P{i + 1}: Hash(KP_New) = {H_new[:16]}...")

    if len(set(hash_values)) == 1:
        print("\n✓ Success! All participants have the same new key")
        return KP_new
    else:
        print("\n✗ Error: Hash values do not match!")
        return None


def elect_leader(num_parties, KP_root=None):
    """
    Election leader: Deterministic election algorithm based on hash values

    Args:
        num_parties: Participants number
        KP_root: Optional root key, used to increase the randomness of the election

    Returns:
        Leader index (0 to num_parties-1)
    """
    candidate_scores = []
    for i in range(num_parties):
        candidate_id = f"P{i + 1}"
        if KP_root:
            election_data = candidate_id + KP_root
        else:
            election_data = candidate_id + str(time.time())

        hash_value = hashlib.sha256(election_data.encode('utf-8')).hexdigest()
        score = int(hash_value, 16)
        candidate_scores.append((i, score, hash_value))

    leader_idx, max_score, leader_hash = max(candidate_scores, key=lambda x: x[1])

    print(f"  Election results:")
    for idx, score, hash_val in candidate_scores:
        marker = " ← Leader" if idx == leader_idx else ""
        print(f"    P{idx + 1}: score={score % 1000000}, hash={hash_val[:16]}...{marker}")

    return leader_idx


# ============================================================================
# QKA Protocol Functions
# ============================================================================

def run_qka_protocol_three(M, num_decoy, num_parties, party_prefix="P", verbose=True, party_names=None):
    """
    Execute complete QKA protocol, using three-particle GHZ-like states

    Args:
        M: GHZ-like state number
        num_decoy: Decoy state number
        num_parties: Participants number
        party_prefix: Participant name prefix
        verbose: Whether to display detailed output
        party_names: Optional list of participant names

    Returns:
        final_keys, all_used_ghz_types, success, key_senders, key_receivers, pair_diffs, announce_diffs
    """
    all_used_ghz_types = set()

    if verbose:
        print(f"\n  === {num_parties}-Party QKA Protocol (Three-particle) ===")
        print("  Preparing quantum states...")

    # Prepare sequences
    states = [
        [create_ghz_like_state_three(random.randint(0, 7)) for _ in range(M)]
        for _ in range(num_parties)
    ]

    # Insert decoy states
    if verbose:
        print("  Inserting decoy states...")
    states_with_decoy, decoy_infos = [], []
    for party_states in states:
        seq_with_decoy, info = insert_single_photons(party_states, num_decoy)
        states_with_decoy.append(seq_with_decoy)
        decoy_infos.append(info)

    # Verify decoy states
    if verbose:
        print("  Verifying decoy states...")
    ok_flags = [
        verify_single_photons(seq, info)
        for seq, info in zip(states_with_decoy, decoy_infos)
    ]
    if not all(ok_flags):
        if verbose:
            print("  Protocol terminated due to high error rate in decoy states")
        return None, set(), False, None, None, None, None

    # Remove decoy states
    if verbose:
        print("  Removing decoy states...")
    states = [
        remove_single_photons(seq, info)
        for seq, info in zip(states_with_decoy, decoy_infos)
    ]

    # Generate keys
    if verbose:
        print("  Generating shared keys...")
    key_senders, key_receivers = [], []
    for i in range(num_parties):
        sender = f"{party_prefix}{i + 1}"
        receiver = f"{party_prefix}{(i + 1) % num_parties + 1}"
        ks, kr, used_types = simulate_pair_three(M, sender, receiver)
        key_senders.append(ks)
        key_receivers.append(kr)
        all_used_ghz_types.update(used_types)

    def get_party_name(idx):
        if party_names and idx < len(party_names):
            return party_names[idx]
        return f"{party_prefix}{idx + 1}"

    if verbose:
        print("\n  Key Information:")
        key_rows = []
        for i, (ks, kr) in enumerate(zip(key_senders, key_receivers)):
            sender = get_party_name(i)
            receiver = get_party_name((i + 1) % num_parties)
            key_rows.append([
                f"K{i + 1} ({sender}→{receiver})",
                ks,
                kr,
                "Match" if ks == kr else "Mismatch"
            ])
        print(format_table(["Key", "Sender", "Receiver", "Status"], key_rows))

    # Find and announce differences
    pair_diffs = [find_different_positions(ks, kr) for ks, kr in zip(key_senders, key_receivers)]
    announce_diffs = [find_different_positions(key_receivers[(i - 1) % num_parties], key_senders[i]) for i in
                      range(num_parties)]

    if verbose:
        print("\n  Announcing differences...")
        adj_rows = []
        for i in range(num_parties):
            party_name = get_party_name(i)
            prev_idx = (i - 1) % num_parties
            comparison = f"K{prev_idx + 1} (in) vs K{i + 1} (out)"
            adj_rows.append([party_name, comparison, str(announce_diffs[i])])
        print(format_table(["Party", "Comparison", "Different Positions"], adj_rows))

    # Calculate final keys
    if verbose:
        print("\n  Calculating final shared keys...")
    final_keys = []
    for i in range(num_parties):
        parts = []
        for j in range(num_parties):
            if i == j:
                parts.append(key_senders[j])
            elif i == (j + 1) % num_parties:
                parts.append(flip_bits(key_receivers[j], pair_diffs[j]))
            else:
                parts.append(key_senders[j])
        final_keys.append(''.join(parts))

    if verbose:
        print("\n  Final Shared Keys:")
        final_rows = [[get_party_name(i), final_keys[i]] for i in range(num_parties)]
        print(format_table(["Party", "Final Key"], final_rows))

    success = len(set(final_keys)) == 1
    if verbose:
        if success:
            print("\n  ✓ Success! All parties share the same final key")
        else:
            print("\n  ✗ Error: Final keys do not match across all parties")

    return final_keys, all_used_ghz_types, success, key_senders, key_receivers, pair_diffs, announce_diffs


def run_qka_protocol_four(M, num_decoy, num_parties, party_prefix="P", verbose=True, party_names=None):
    """
    Execute the complete multi-party QKA protocol, using four-particle GHZ-like states

    Args:
        M: GHZ-like state number
        num_decoy: Decoy state number
        num_parties: Participants number
        party_prefix: Participant name prefix
        verbose: Whether to display detailed output
        party_names: Optional list of participant names

    Returns:
        final_keys, all_used_ghz_types, success, keys, diff_lists
    """
    all_used_ghz_types = set()

    if verbose:
        print(f"\n  === {num_parties}-Party QKA Protocol (Four-particle) ===")
        print("  Preparing quantum states...")
        print("  Inserting decoy states...")

    keys = []
    error_rates = []
    decoy_error_counts = []
    decoy_totals = []

    def get_party_name(idx):
        if party_names and idx < len(party_names):
            return party_names[idx]
        return f"{party_prefix}{idx + 1}"

    for i in range(num_parties):
        sender = get_party_name(i)
        receiver = get_party_name((i + 1) % num_parties)
        key, error_rate, _, err_cnt, err_total, used_types = distribute_key_between_participants_four(sender, receiver,
                                                                                                      M, num_decoy)
        while error_rate is None or error_rate > 0.15:
            key, error_rate, _, err_cnt, err_total, used_types = distribute_key_between_participants_four(sender,
                                                                                                          receiver, M,
                                                                                                          num_decoy)
        keys.append(key)
        error_rates.append(error_rate)
        decoy_error_counts.append(err_cnt)
        decoy_totals.append(err_total)
        if used_types:
            all_used_ghz_types.update(used_types)

    if verbose:
        print("  Verifying decoy states...")
        for i in range(num_parties):
            print(f"  Decoy state error rate: {error_rates[i]:.2%} ({decoy_error_counts[i]}/{decoy_totals[i]} errors)")
        print("  Removing decoy states...")
        print("  Generating shared keys...")

        print("\n  Key Information:")
        key_rows = []
        for i, key in enumerate(keys):
            sender = get_party_name(i)
            receiver = get_party_name((i + 1) % num_parties)
            key_rows.append([f"K{i + 1} ({sender}→{receiver})", key, key, "Match"])
        print(format_table(["Key", "Sender", "Receiver", "Status"], key_rows))

    # Find and announce differences
    diff_lists = []
    for i in range(num_parties):
        incoming_idx = (i - 1) % num_parties
        diffs = find_different_positions(keys[incoming_idx], keys[i])
        diff_lists.append(diffs)

    if verbose:
        print("\n  Announcing differences...")
        diff_rows = []
        for i in range(num_parties):
            party_name = get_party_name(i)
            incoming_idx = (i - 1) % num_parties
            comparison = f"K{incoming_idx + 1} (in) vs K{i + 1} (out)"
            diff_rows.append([party_name, comparison, str(diff_lists[i])])
        print(format_table(["Participant", "Comparison", "Different Positions"], diff_rows))

    # Calculate final keys
    if verbose:
        print("\n  Calculating final shared keys...")
    final_key = ''.join(keys)
    final_keys = [final_key] * num_parties

    if verbose:
        print("\n  Final Shared Keys:")
        final_rows = [[get_party_name(i), final_keys[i]] for i in range(num_parties)]
        print(format_table(["Party", "Final Key"], final_rows))

    success = len(set(final_keys)) == 1
    if verbose:
        if success:
            print("\n  ✓ Success! All parties share the same final key")
        else:
            print("\n  ✗ Error: Final keys do not match across all parties")

    return final_keys, all_used_ghz_types, success, keys, diff_lists


# ============================================================================
# Scene 2 and Scene 3 Functions (Common, but call different QKA protocols)
# ============================================================================

def scene2_new_members_join(existing_parties, new_members_count, KP_root, M, num_decoy, particle_type):
    """
    Scene 2: Key Distribution Mechanism when only new members join

    Args:
        existing_parties: Existing participants number
        new_members_count: New participants number
        KP_root: Root key (original shared key)
        M: GHZ-like state number
        num_decoy: Decoy state number
        particle_type: 'three' or 'four' - which particle type to use

    Returns:
        success, final_key, used_ghz_types
    """
    print("\n=== Scene 2: New Members Join (Key Distribution) ===")
    print(f"Existing participants: {existing_parties}")
    print(f"New participants: {new_members_count}")
    print(f"Root key (KP): {KP_root}")

    print("\nStep 1: Electing leader...")
    leader_idx = elect_leader(existing_parties, KP_root)
    print(f"  Leader elected: P{leader_idx + 1}")

    print("\nStep 2: Leader and new members executing QKA to get K_Group...")

    group_parties = 1 + new_members_count
    target_key_length = len(KP_root)
    M_group = math.ceil(target_key_length / (group_parties * 2))
    print(f"  Target key length: {target_key_length} bits")
    print(f"  Group participants: {group_parties} (1 leader + {new_members_count} new members)")
    print(f"  Using M={M_group} GHZ-like states per party")
    print(f"  Using {num_decoy} decoy states per party")
    print(f"  Using {particle_type}-particle GHZ-like states for QKA...")

    group_party_names = [f"P{leader_idx + 1} (Leader)"] + [f"P_New{i + 1}" for i in range(new_members_count)]

    # Execute QKA protocol based on particle type
    if particle_type == 'three':
        final_keys_group, used_ghz_types, qka_success, key_senders_group, key_receivers_group, pair_diffs_group, announce_diffs_group = run_qka_protocol_three(
            M_group, num_decoy, group_parties, party_prefix="P_Group", verbose=True, party_names=group_party_names
        )
    else:  # four
        final_keys_group, used_ghz_types, qka_success, keys_group, diff_lists_group = run_qka_protocol_four(
            M_group, num_decoy, group_parties, party_prefix="P_Group", verbose=True, party_names=group_party_names
        )

    if not qka_success or not final_keys_group:
        print("  Error: QKA protocol failed for leader and new members")
        return False, None, set()

    K_group = final_keys_group[0]
    if len(K_group) > target_key_length:
        K_group = K_group[:target_key_length]
    elif len(K_group) < target_key_length:
        repeat_times = (target_key_length // len(K_group)) + 1
        K_group = (K_group * repeat_times)[:target_key_length]

    print(f"\n  Group key K_Group generated (length: {len(K_group)} bits)")
    print(f"  All {group_parties} participants in the group share the same key")

    print("\nStep 3: Leader announcing differences between K_Group and KP...")
    KP_root_aligned = KP_root

    diff_positions = find_different_positions(K_group, KP_root_aligned)
    if len(diff_positions) > 20:
        print(f"  Different positions (first 20): {diff_positions[:20]}...")
        print(f"  Total different positions: {len(diff_positions)}")
    else:
        print(f"  Different positions: {diff_positions}")

    print("\nStep 4: New members adjusting keys based on differences...")
    adjusted_keys = []
    for w in range(new_members_count):
        adjusted_key = flip_bits(K_group, diff_positions)
        adjusted_keys.append(adjusted_key)

    all_keys = [KP_root_aligned] * existing_parties + adjusted_keys
    if len(set(all_keys)) == 1:
        if new_members_count == 1:
            new_participants_str = "P_New1"
        else:
            new_participants_str = f"P_New1-P_New{new_members_count}"
        print(f"  The shared key for new participants {new_participants_str} after adjustment: {all_keys[0]}")
        print("\n✓ Success! All participants (including new members) share the same key")
        return True, all_keys[0], used_ghz_types
    else:
        print("\n✗ Error: Keys do not match across all participants")
        for i, k in enumerate(all_keys):
            party_name = f"P{i + 1}" if i < existing_parties else f"P_New{i - existing_parties + 1}"
            print(f"  {party_name}: {k[:50]}..." if len(k) > 50 else f"  {party_name}: {k}")
        return False, None, used_ghz_types


def scene3_composite_scenario(initial_parties, leaving_count, joining_count, KP_root, M, num_decoy, particle_type):
    """
    Scene 3: Composite Scenario (Leaving + Joining)

    Args:
        initial_parties: Initial participants number
        leaving_count: Leaving participants number
        joining_count: Joining participants number
        KP_root: Root key
        M: GHZ-like state number
        num_decoy: Decoy state number
        particle_type: 'three' or 'four' - which particle type to use

    Returns:
        KP_new, used_ghz_types
    """
    print("\n=== Scene 3: Composite Scenario (Leaving + Joining) ===")
    print(f"Initial participants: {initial_parties}")
    print(f"Leaving participants: {leaving_count}")
    print(f"Joining participants: {joining_count}")

    print("\n--- Phase 1: New Members Join (Scene 2) ---")
    remaining_parties = initial_parties - leaving_count
    success, shared_key, scene2_ghz_types = scene2_new_members_join(remaining_parties, joining_count, KP_root, M,
                                                                    num_decoy, particle_type)

    if not success:
        print("Phase 1 failed, aborting Scene 3")
        return None, scene2_ghz_types if 'scene2_ghz_types' in locals() else set()

    print("\n--- Phase 2: Key Update (Scene 1) ---")
    total_parties = remaining_parties + joining_count
    KP_new = scene1_key_update(total_parties, shared_key)

    if KP_new:
        print("\n✓ Success! Scene 3 completed successfully")
        return KP_new, scene2_ghz_types
    else:
        print("\n✗ Error: Scene 3 Phase 2 failed")
        return None, scene2_ghz_types


# ============================================================================
# Circuit Visualization Functions
# ============================================================================

def build_ghz_circuit_with_measurements_three(type_num):
    """Construct a complete circuit with Z measurement and Bell measurement for three-particle GHZ-like state"""
    qc = create_ghz_like_state_three(type_num)
    z_measure(qc)
    bell_measure(qc)
    qc.name = f"GHZ-like type {type_num}"
    return qc


def create_circuit_with_measurement_four(state_type):
    """Create a complete circuit with four-particle GHZ state preparation and measurement"""
    qreg = QuantumRegister(4, 'q')
    creg = ClassicalRegister(4, 'c')
    qc = QuantumCircuit(qreg, creg)

    prepare_ghz_like_four(qc, qreg, state_type)
    qc.barrier()

    # Bell measurement: particles 1 and 2
    qc.cx(qreg[0], qreg[1])
    qc.h(qreg[0])
    qc.measure(qreg[0], creg[0])
    qc.measure(qreg[1], creg[1])

    # Bell measurement: particles 3 and 4
    qc.cx(qreg[2], qreg[3])
    qc.h(qreg[2])
    qc.measure(qreg[2], creg[2])
    qc.measure(qreg[3], creg[3])

    return qc


def visualize_ghz_circuits(used_types, particle_type):
    """
    Generate and display comprehensive circuit diagrams for the actually used GHZ-like states.

    Args:
        used_types: A set or list of GHZ-like state types that were actually used
        particle_type: 'three' or 'four' - which particle type to visualize
    """
    if not used_types:
        print("No GHZ-like states were used, skipping circuit diagram generation.")
        return

    sorted_types = sorted(used_types)
    num_types = len(sorted_types)
    cols = 2
    rows = (num_types + cols - 1) // cols

    if particle_type == 'three':
        # Three-particle visualization
        fig, axes = plt.subplots(rows, cols, figsize=(12, 10), squeeze=False)
        axes = axes.flatten()

        for idx, type_num in enumerate(sorted_types):
            qc = build_ghz_circuit_with_measurements_three(type_num)
            ax = axes[idx]
            qc.draw("mpl", ax=ax)
            ax.set_title(r"$G_{%d}$" % type_num, fontsize=12)

        for idx in range(num_types, len(axes)):
            fig.delaxes(axes[idx])

        type_range_str = f"G{min(sorted_types)}-G{max(sorted_types)}" if len(
            sorted_types) > 1 else f"G{sorted_types[0]}"
        fig.suptitle(f"Three-particle GHZ-like States with Measurements ({type_range_str})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    else:  # four-particle
        # Split used types into G0-G7 and G8-G15
        types_g0_g7 = sorted([t for t in used_types if 0 <= t <= 7])
        types_g8_g15 = sorted([t for t in used_types if 8 <= t <= 15])

        # Generate first figure for G0-G7 if any
        if types_g0_g7:
            print("Generating first comprehensive circuit diagram (G0-G7)...")
            num_types = len(types_g0_g7)
            rows = (num_types + cols - 1) // cols

            fig1, axes1 = plt.subplots(rows, cols, figsize=(20, 14), squeeze=False)
            axes1 = axes1.flatten()

            for idx, type_num in enumerate(types_g0_g7):
                qc = create_circuit_with_measurement_four(type_num)
                ax = axes1[idx]
                qc.draw(output='mpl', style='clifford',
                        initial_state=False,
                        cregbundle=True,
                        plot_barriers=True,
                        ax=ax)
                ax.set_title(r"$G_{%d}$" % type_num, fontsize=12)

            for idx in range(num_types, len(axes1)):
                fig1.delaxes(axes1[idx])

            type_range_str = f"G{min(types_g0_g7)}-G{max(types_g0_g7)}" if len(
                types_g0_g7) > 1 else f"G{types_g0_g7[0]}"
            fig1.suptitle(f"Four-particle GHZ-like States with Measurements ({type_range_str})", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        # Generate second figure for G8-G15 if any
        if types_g8_g15:
            print("Generating second comprehensive circuit diagram (G8-G15)...")
            num_types = len(types_g8_g15)
            rows = (num_types + cols - 1) // cols

            fig2, axes2 = plt.subplots(rows, cols, figsize=(20, 14), squeeze=False)
            axes2 = axes2.flatten()

            for idx, type_num in enumerate(types_g8_g15):
                qc = create_circuit_with_measurement_four(type_num)
                ax = axes2[idx]
                qc.draw(output='mpl', style='clifford',
                        initial_state=False,
                        cregbundle=True,
                        plot_barriers=True,
                        ax=ax)
                ax.set_title(r"$G_{%d}$" % type_num, fontsize=12)

            for idx in range(num_types, len(axes2)):
                fig2.delaxes(axes2[idx])

            type_range_str = f"G{min(types_g8_g15)}-G{max(types_g8_g15)}" if len(
                types_g8_g15) > 1 else f"G{types_g8_g15[0]}"
            fig2.suptitle(f"Four-particle GHZ-like States with Measurements ({type_range_str})", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    print("All circuit diagrams have been generated and displayed!")


def visualize_ghz_measurement_results(used_types, particle_type):
    """
    Generate and display measurement results (1024 shots) for the actually used GHZ-like states.

    Args:
        used_types: A set or list of GHZ-like state types that were actually used
        particle_type: 'three' or 'four' - which particle type to visualize
    """
    if not used_types:
        print("No GHZ-like states were used, skipping measurement results diagram generation.")
        return

    sorted_types = sorted(used_types)
    cols = 2
    max_rows_per_fig = 4

    if particle_type == 'three':
        # Three-particle visualization
        num_types = len(sorted_types)
        rows = min((num_types + cols - 1) // cols, max_rows_per_fig)

        types_per_fig = cols * max_rows_per_fig
        num_figures = (num_types + types_per_fig - 1) // types_per_fig

        for fig_idx in range(num_figures):
            start_idx = fig_idx * types_per_fig
            end_idx = min(start_idx + types_per_fig, num_types)
            fig_types = sorted_types[start_idx:end_idx]
            num_fig_types = len(fig_types)
            rows = min((num_fig_types + cols - 1) // cols, max_rows_per_fig)

            fig, axes = plt.subplots(rows, cols, figsize=(14, 12), squeeze=False)
            axes = axes.flatten()

            for idx, type_num in enumerate(fig_types):
                # Create circuit and run 1024 shots
                qc = build_ghz_circuit_with_measurements_three(type_num)
                compiled = transpile(qc, backend)
                result = backend.run(compiled, shots=1024).result()
                counts = result.get_counts()

                # Prepare data for bar chart (three-particle: 3 bits = 8 outcomes)
                all_possible_outcomes = [f"{i:03b}" for i in range(8)]  # 000 to 111

                outcome_counts = []
                for outcome in all_possible_outcomes:
                    # Qiskit returns results in reverse order, so we need to reverse
                    qiskit_outcome = outcome[::-1]
                    count = counts.get(qiskit_outcome, 0)
                    outcome_counts.append(count)

                # Plot bar chart
                ax = axes[idx]
                bars = ax.bar(range(len(all_possible_outcomes)), outcome_counts, alpha=0.7)
                ax.set_xlabel('Measurement Outcome', fontsize=10)
                ax.set_ylabel('Count (1024 shots)', fontsize=10)
                ax.set_title(r"$G_{%d}$" % type_num, fontsize=12, fontweight='bold')
                ax.set_xticks(range(len(all_possible_outcomes)))
                ax.set_xticklabels(all_possible_outcomes, rotation=45, ha='right', fontsize=8)
                ax.grid(axis='y', alpha=0.3)
                ax.set_ylim(0, max(outcome_counts) * 1.1 if outcome_counts else 1)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', fontsize=7)

            # Hide unused axes
            for idx in range(num_fig_types, len(axes)):
                fig.delaxes(axes[idx])

            type_range_str = f"G{min(fig_types)}-G{max(fig_types)}" if len(fig_types) > 1 else f"G{fig_types[0]}"
            fig.suptitle(f"Three-particle GHZ-like States Measurement Results (1024 shots) ({type_range_str})",
                         fontsize=14, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

    else:  # four-particle
        # Split used types into G0-G7 and G8-G15
        types_g0_g7 = sorted([t for t in used_types if 0 <= t <= 7])
        types_g8_g15 = sorted([t for t in used_types if 8 <= t <= 15])

        # Generate first figure for G0-G7 if any
        if types_g0_g7:
            print("Generating first measurement results diagram (G0-G7)...")
            num_types = len(types_g0_g7)
            rows = min((num_types + cols - 1) // cols, max_rows_per_fig)

            types_per_fig = cols * max_rows_per_fig
            num_figures = (num_types + types_per_fig - 1) // types_per_fig

            for fig_idx in range(num_figures):
                start_idx = fig_idx * types_per_fig
                end_idx = min(start_idx + types_per_fig, num_types)
                fig_types = types_g0_g7[start_idx:end_idx]
                num_fig_types = len(fig_types)
                rows = min((num_fig_types + cols - 1) // cols, max_rows_per_fig)

                fig1, axes1 = plt.subplots(rows, cols, figsize=(16, 12), squeeze=False)
                axes1 = axes1.flatten()

                for idx, type_num in enumerate(fig_types):
                    # Create circuit and run 1024 shots
                    qc = create_circuit_with_measurement_four(type_num)
                    compiled = transpile(qc, backend)
                    result = backend.run(compiled, shots=1024).result()
                    counts = result.get_counts()

                    # Prepare data for bar chart
                    all_possible_outcomes = [f"{i:04b}" for i in range(16)]  # 0000 to 1111

                    outcome_counts = []
                    for outcome in all_possible_outcomes:
                        # Qiskit returns results in reverse order
                        qiskit_outcome = outcome[::-1]
                        count = counts.get(qiskit_outcome, 0)
                        outcome_counts.append(count)

                    # Plot bar chart
                    ax = axes1[idx]
                    bars = ax.bar(range(len(all_possible_outcomes)), outcome_counts, alpha=0.7)
                    ax.set_xlabel('Measurement Outcome', fontsize=10)
                    ax.set_ylabel('Count (1024 shots)', fontsize=10)
                    ax.set_title(r"$G_{%d}$" % type_num, fontsize=12, fontweight='bold')
                    ax.set_xticks(range(len(all_possible_outcomes)))
                    ax.set_xticklabels(all_possible_outcomes, rotation=45, ha='right', fontsize=7)
                    ax.grid(axis='y', alpha=0.3)
                    ax.set_ylim(0, max(outcome_counts) * 1.1 if outcome_counts else 1)

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', fontsize=6)

                # Hide unused axes
                for idx in range(num_fig_types, len(axes1)):
                    fig1.delaxes(axes1[idx])

                type_range_str = f"G{min(fig_types)}-G{max(fig_types)}" if len(fig_types) > 1 else f"G{fig_types[0]}"
                fig1.suptitle(f"Four-particle GHZ-like States Measurement Results (1024 shots) ({type_range_str})",
                              fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

        # Generate second figure for G8-G15 if any
        if types_g8_g15:
            print("Generating second measurement results diagram (G8-G15)...")
            num_types = len(types_g8_g15)
            rows = min((num_types + cols - 1) // cols, max_rows_per_fig)

            types_per_fig = cols * max_rows_per_fig
            num_figures = (num_types + types_per_fig - 1) // types_per_fig

            for fig_idx in range(num_figures):
                start_idx = fig_idx * types_per_fig
                end_idx = min(start_idx + types_per_fig, num_types)
                fig_types = types_g8_g15[start_idx:end_idx]
                num_fig_types = len(fig_types)
                rows = min((num_fig_types + cols - 1) // cols, max_rows_per_fig)

                fig2, axes2 = plt.subplots(rows, cols, figsize=(16, 12), squeeze=False)
                axes2 = axes2.flatten()

                for idx, type_num in enumerate(fig_types):
                    # Create circuit and run 1024 shots
                    qc = create_circuit_with_measurement_four(type_num)
                    compiled = transpile(qc, backend)
                    result = backend.run(compiled, shots=1024).result()
                    counts = result.get_counts()

                    # Prepare data for bar chart
                    all_possible_outcomes = [f"{i:04b}" for i in range(16)]  # 0000 to 1111

                    outcome_counts = []
                    for outcome in all_possible_outcomes:
                        # Qiskit returns results in reverse order
                        qiskit_outcome = outcome[::-1]
                        count = counts.get(qiskit_outcome, 0)
                        outcome_counts.append(count)

                    # Plot bar chart
                    ax = axes2[idx]
                    bars = ax.bar(range(len(all_possible_outcomes)), outcome_counts, alpha=0.7)
                    ax.set_xlabel('Measurement Outcome', fontsize=10)
                    ax.set_ylabel('Count (1024 shots)', fontsize=10)
                    ax.set_title(r"$G_{%d}$" % type_num, fontsize=12, fontweight='bold')
                    ax.set_xticks(range(len(all_possible_outcomes)))
                    ax.set_xticklabels(all_possible_outcomes, rotation=45, ha='right', fontsize=7)
                    ax.grid(axis='y', alpha=0.3)
                    ax.set_ylim(0, max(outcome_counts) * 1.1 if outcome_counts else 1)

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{int(height)}',
                                    ha='center', va='bottom', fontsize=6)

                # Hide unused axes
                for idx in range(num_fig_types, len(axes2)):
                    fig2.delaxes(axes2[idx])

                type_range_str = f"G{min(fig_types)}-G{max(fig_types)}" if len(fig_types) > 1 else f"G{fig_types[0]}"
                fig2.suptitle(f"Four-particle GHZ-like States Measurement Results (1024 shots) ({type_range_str})",
                              fontsize=14, fontweight='bold')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.show()

    print("All measurement results diagrams have been generated and displayed!")


# ============================================================================
# Main Protocol Functions
# ============================================================================

def run_n_party_protocol_three(M, num_decoy, num_parties):
    """Run the complete N-party protocol using three-particle GHZ-like states"""
    t_start = time.perf_counter()
    all_used_ghz_types = set()
    print(f"\n=== {num_parties}-Party Quantum Key Distribution Protocol (Three-particle) ===")

    print("\n1. Preparing quantum states...")
    states = [
        [create_ghz_like_state_three(random.randint(0, 7)) for _ in range(M)]
        for _ in range(num_parties)
    ]

    print("\n2. Inserting decoy states...")
    states_with_decoy, decoy_infos = [], []
    for party_states in states:
        seq_with_decoy, info = insert_single_photons(party_states, num_decoy)
        states_with_decoy.append(seq_with_decoy)
        decoy_infos.append(info)

    print("\n3. Verifying decoy states...")
    ok_flags = [
        verify_single_photons(seq, info)
        for seq, info in zip(states_with_decoy, decoy_infos)
    ]
    if not all(ok_flags):
        print("\nProtocol terminated due to high error rate in decoy states")
        return

    print("\n4. Removing decoy states...")
    states = [
        remove_single_photons(seq, info)
        for seq, info in zip(states_with_decoy, decoy_infos)
    ]

    print("\n5. Generating shared keys...")
    key_senders, key_receivers = [], []
    for i in range(num_parties):
        sender = f"P{i + 1}"
        receiver = f"P{(i + 1) % num_parties + 1}"
        ks, kr, used_types = simulate_pair_three(M, sender, receiver)
        key_senders.append(ks)
        key_receivers.append(kr)
        all_used_ghz_types.update(used_types)

    print("\n6. Key Information:")
    key_rows = []
    for i, (ks, kr) in enumerate(zip(key_senders, key_receivers)):
        sender = f"P{i + 1}"
        receiver = f"P{(i + 1) % num_parties + 1}"
        key_rows.append([
            f"K{i + 1} ({sender}→{receiver})",
            ks,
            kr,
            "Match" if ks == kr else "Mismatch"
        ])
    print(format_table(["Key", "Sender", "Receiver", "Status"], key_rows))

    print("\n7. Announcing differences...")
    pair_diffs = [find_different_positions(ks, kr) for ks, kr in zip(key_senders, key_receivers)]
    announce_diffs = [find_different_positions(key_receivers[(i - 1) % num_parties], key_senders[i]) for i in
                      range(num_parties)]
    adj_rows = []
    for i in range(num_parties):
        prev_idx = (i - 1) % num_parties
        sender = f"P{i + 1}"
        comparison = f"K{prev_idx + 1} (in) vs K{i + 1} (out)"
        adj_rows.append([sender, comparison, str(announce_diffs[i])])
    print(format_table(["Party", "Comparison", "Different Positions"], adj_rows))

    print("\n8. Calculating final shared keys...")
    final_keys = []
    for i in range(num_parties):
        parts = []
        for j in range(num_parties):
            if i == j:
                parts.append(key_senders[j])
            elif i == (j + 1) % num_parties:
                parts.append(flip_bits(key_receivers[j], pair_diffs[j]))
            else:
                parts.append(key_senders[j])
        final_keys.append(''.join(parts))

    print("\n9. Final Shared Keys:")
    final_rows = [[f"P{i + 1}", final_keys[i]] for i in range(num_parties)]
    print(format_table(["Party", "Final Key"], final_rows))

    if len(set(final_keys)) == 1:
        print("\nSuccess! All parties share the same final key")
    else:
        print("\nError: Final keys do not match across all parties")

    key = final_keys[0] if final_keys else ""
    key_length = len(key)
    num_zeros = key.count('0')
    num_ones = key.count('1')
    p0 = num_zeros / key_length if key_length else 0
    p1 = num_ones / key_length if key_length else 0
    entropy = 0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * math.log2(p)
    latency = time.perf_counter() - t_start
    key_rate_bps = key_length / latency if latency > 0 else 0
    classical_cost = 2 * num_parties
    decoy_bits = sum(len(info) * 2 for info in decoy_infos)
    decoy_results_bits = sum(len(info) for info in decoy_infos)
    diff_bits = sum(len(d) for d in announce_diffs)
    classical_bits = decoy_bits + decoy_results_bits + diff_bits
    bit_flips = diff_bits
    quantum_qubits = num_parties * M * 3 + sum(len(info) for info in decoy_infos)

    print("\n==============================================")
    print("Final Key Statistical Analysis")
    print("==============================================")
    stats_rows = [
        ["Final key", key],
        ["Key length", f"{key_length} bits"],
        ["Number of 0s", num_zeros],
        ["Number of 1s", num_ones],
        ["Proportion of 0s", f"{p0:.2%}"],
        ["Proportion of 1s", f"{p1:.2%}"],
        ["Shannon entropy", f"{entropy:.4f} bits"],
        ["Key rate", f"{key_rate_bps:.4f} bit/s"],
        ["Latency", f"{latency:.4f} s"],
        ["Quantum resource cost", f"{quantum_qubits} qubits"],
        ["Classical resource cost", f"{classical_cost} messages, {classical_bits} classical bits"],
        ["Bit flips (key adjustment)", f"{bit_flips}"],
    ]
    print(format_table(["Statistic", "Value"], stats_rows))

    # Dynamic Adaptive Key Update Mechanism
    print("\n" + "=" * 60)
    print("Dynamic Adaptive Key Update Mechanism")
    print("=" * 60)

    scene = random.choice([1, 2, 3])
    KP_root = key

    final_key_used = KP_root
    scenario_name = "Initial QKA Only"
    final_participants = num_parties
    Q_val = 0
    W_val = 0

    if scene == 1:
        print("\n[Scene 1: Only Members Withdraw]")
        print("Please enter the number of leaving participants Q, where Q<n")
        Q = int(input("Enter number of leaving participants (Q): "))

        if Q >= num_parties:
            print(f"Error: Q ({Q}) must be less than n ({num_parties})")
            return

        remaining_parties = num_parties - Q
        print(f"\nParticipants leaving: {Q}")
        print(f"Remaining participants: {remaining_parties}")

        KP_new = scene1_key_update(remaining_parties, KP_root)
        Q_val = Q

        if KP_new:
            final_key_used = KP_new
            scenario_name = "Scene 1: Participants Withdrawal"
            final_participants = num_parties - Q_val
            print("\n" + "=" * 60)
            print("Key Update Summary")
            print("=" * 60)
            print(f"Old key (KP_Old): {KP_root}")
            print(f"New key (KP_New): {KP_new}")
            print(f"Key length: {len(KP_new)} bits")
            print("=" * 60)
        else:
            print("\nKey update failed!")

    elif scene == 2:
        print("\n[Scene 2: Only New Members Join]")
        print("Please enter the number of new participants W.")
        W = int(input("Enter number of new participants (W): "))

        if W <= 0:
            print("Error: W must be greater than 0")
            return

        print(f"\nNew participants joining: {W}")

        success, final_key, scene2_ghz_types = scene2_new_members_join(num_parties, W, KP_root, M, num_decoy, 'three')
        all_used_ghz_types.update(scene2_ghz_types)
        W_val = W

        if success:
            final_key_used = final_key
            scenario_name = "Scene 2: New Members Join"
            final_participants = num_parties + W_val
            print("\n" + "=" * 60)
            print("Key Distribution Summary")
            print("=" * 60)
            print(f"Root key (KP): {KP_root}")
            print(f"Final shared key: {final_key}")
            print(f"Total participants: {num_parties + W}")
            print(f"Key length: {len(final_key)} bits")
            print("=" * 60)
        else:
            print("\nKey distribution failed!")

    elif scene == 3:
        print("\n[Scene 3: Participants Withdrawal and New Members Join]")
        print("Please enter the number of leaving participants Q, and the number of new participants W")
        Q = int(input("Enter number of leaving participants (Q): "))
        W = int(input("Enter number of new participants (W): "))

        if Q >= num_parties:
            print(f"Error: Q ({Q}) must be less than n ({num_parties})")
            return

        if W <= 0:
            print("Error: W must be greater than 0")
            return

        print(f"\nParticipants leaving: {Q}")
        print(f"New participants joining: {W}")
        print(f"Remaining original participants: {num_parties - Q}")
        print(f"Total participants after update: {num_parties - Q + W}")

        KP_new, scene3_ghz_types = scene3_composite_scenario(num_parties, Q, W, KP_root, M, num_decoy, 'three')
        all_used_ghz_types.update(scene3_ghz_types)
        Q_val = Q
        W_val = W

        if KP_new:
            final_key_used = KP_new
            scenario_name = "Scene 3: Composite Scenario"
            final_participants = num_parties - Q_val + W_val
            print("\n" + "=" * 60)
            print("Composite Scenario Summary")
            print("=" * 60)
            print(f"Initial root key (KP): {KP_root}")
            print(f"Final new key (KP_New): {KP_new}")
            print(f"Key length: {len(KP_new)} bits")
            print("=" * 60)
        else:
            print("\nComposite scenario execution failed!")

    # Overall Statistical Analysis
    t_end = time.perf_counter()
    total_latency = t_end - t_start

    final_key_length = len(final_key_used) if final_key_used else 0
    num_zeros_final = final_key_used.count('0') if final_key_used else 0
    num_ones_final = final_key_used.count('1') if final_key_used else 0
    p0_final = num_zeros_final / final_key_length if final_key_length > 0 else 0
    p1_final = num_ones_final / final_key_length if final_key_length > 0 else 0
    entropy_final = 0
    for p in (p0_final, p1_final):
        if p > 0:
            entropy_final -= p * math.log2(p)

    initial_quantum_qubits = num_parties * M * 3 + sum(len(info) for info in decoy_infos)
    initial_classical_cost = 2 * num_parties
    initial_decoy_bits = sum(len(info) * 2 for info in decoy_infos)
    initial_decoy_results_bits = sum(len(info) for info in decoy_infos)
    initial_diff_bits = sum(len(d) for d in announce_diffs)
    initial_classical_bits = initial_decoy_bits + initial_decoy_results_bits + initial_diff_bits

    additional_quantum_qubits = 0
    additional_classical_bits = 0
    additional_messages = 0

    if scene == 2:
        group_parties = 1 + W_val
        M_group = math.ceil(len(KP_root) / (group_parties * 2)) if len(KP_root) > 0 else M
        additional_quantum_qubits = group_parties * M_group * 3 + group_parties * num_decoy
        additional_classical_bits = group_parties * num_decoy * 2 + group_parties * num_decoy
        additional_messages = group_parties * 2
    elif scene == 3:
        group_parties = 1 + W_val
        M_group = math.ceil(len(KP_root) / (group_parties * 2)) if len(KP_root) > 0 else M
        additional_quantum_qubits = group_parties * M_group * 3 + group_parties * num_decoy
        additional_classical_bits = group_parties * num_decoy * 2 + group_parties * num_decoy
        additional_messages = group_parties * 2
        remaining_parties = num_parties - Q_val
        additional_classical_bits += remaining_parties * 32 * 2
        additional_messages += remaining_parties * 2
    elif scene == 1:
        remaining_parties = num_parties - Q_val
        additional_classical_bits = remaining_parties * 32 * 2
        additional_messages = remaining_parties * 2

    total_quantum_qubits = initial_quantum_qubits + additional_quantum_qubits
    total_classical_bits = initial_classical_bits + additional_classical_bits
    total_messages = initial_classical_cost + additional_messages

    overall_key_rate = final_key_length / total_latency if total_latency > 0 else 0
    ghz_types_formatted = ", ".join([f"G{i}" for i in sorted(all_used_ghz_types)])

    print("\n" + "=" * 60)
    print("Overall Program Statistical Analysis")
    print("=" * 60)
    overall_stats_rows = [
        ["Protocol scenario", scenario_name],
        ["Final key", final_key_used if final_key_used else "N/A"],
        ["Final key length", f"{final_key_length} bits"],
        ["Number of 0s", num_zeros_final],
        ["Number of 1s", num_ones_final],
        ["Proportion of 0s", f"{p0_final:.2%}"],
        ["Proportion of 1s", f"{p1_final:.2%}"],
        ["Shannon entropy", f"{entropy_final:.4f} bits"],
        ["Overall key rate", f"{overall_key_rate:.4f} bit/s"],
        ["Total latency", f"{total_latency:.4f} s"],
        ["Total quantum resource cost", f"{total_quantum_qubits} qubits"],
        ["  - Initial QKA", f"{initial_quantum_qubits} qubits"],
        ["  - Dynamic scenarios", f"{additional_quantum_qubits} qubits"],
        ["Total classical resource cost", f"{total_messages} messages, {total_classical_bits} classical bits"],
        ["  - Initial QKA", f"{initial_classical_cost} messages, {initial_classical_bits} classical bits"],
        ["  - Dynamic scenarios", f"{additional_messages} messages, {additional_classical_bits} classical bits"],
        ["Number of GHZ-like state types used", f"{len(all_used_ghz_types)} types: {ghz_types_formatted}"],
        ["Initial participants", f"{num_parties}"],
    ]

    if scene == 1:
        overall_stats_rows.append(["Participants leaving", f"{Q_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    elif scene == 2:
        overall_stats_rows.append(["Participants joining", f"{W_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    elif scene == 3:
        overall_stats_rows.append(["Participants leaving", f"{Q_val}"])
        overall_stats_rows.append(["Participants joining", f"{W_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    else:
        overall_stats_rows.append(["Final participants", f"{final_participants}"])

    print(format_table(["Statistic", "Value"], overall_stats_rows))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Circuit Diagrams for All Used GHZ-like States")
    print("=" * 60)

    if all_used_ghz_types:
        sorted_ghz_types = sorted(all_used_ghz_types)
        ghz_types_list = [f"G{i}" for i in sorted_ghz_types]
        print(f"\nTotal GHZ-like state types used in this protocol execution: {len(all_used_ghz_types)}")
        print(f"Types: {', '.join(ghz_types_list)}")
        print(f"\nGenerating circuit diagrams for used GHZ-like states...")
        visualize_ghz_circuits(all_used_ghz_types, 'three')

        print("\n" + "=" * 60)
        print("Measurement Results for All Used GHZ-like States (1024 shots)")
        print("=" * 60)
        print(f"\nGenerating measurement results diagrams for used GHZ-like states...")
        visualize_ghz_measurement_results(all_used_ghz_types, 'three')
    else:
        print("\nNo GHZ-like states were used in this protocol execution.")
        print("(This may occur if only Scene 1 was executed, which uses hash-based key update without GHZ states)")


def run_n_party_protocol_four(M, num_decoy, num_parties):
    """Run the complete N-party protocol using four-particle GHZ-like states"""
    t_start = time.perf_counter()
    all_used_ghz_types = set()
    print(f"\n=== {num_parties}-Party Quantum Key Distribution Protocol (Four-particle) ===")
    print("\n1. Preparing quantum states...")
    print("\n2. Inserting decoy states...")

    keys = []
    error_rates = []
    decoy_error_counts = []
    decoy_totals = []

    for i in range(num_parties):
        sender = f"P{i + 1}"
        receiver = f"P{(i + 1) % num_parties + 1}"
        key, error_rate, _, err_cnt, err_total, used_types = distribute_key_between_participants_four(sender, receiver,
                                                                                                      M, num_decoy)
        while error_rate is None or error_rate > 0.15:
            key, error_rate, _, err_cnt, err_total, used_types = distribute_key_between_participants_four(sender,
                                                                                                          receiver, M,
                                                                                                          num_decoy)
        keys.append(key)
        error_rates.append(error_rate)
        decoy_error_counts.append(err_cnt)
        decoy_totals.append(err_total)
        if used_types:
            all_used_ghz_types.update(used_types)

    print("\n3. Verifying decoy states...")
    for i in range(num_parties):
        print(f"Decoy state error rate: {error_rates[i]:.2%} ({decoy_error_counts[i]}/{decoy_totals[i]} errors)")

    print("\n4. Removing decoy states...")
    print("\n5. Generating shared keys...")

    print("\n6. Key Information:")
    key_rows = []
    for i, key in enumerate(keys):
        sender = f"P{i + 1}"
        receiver = f"P{(i + 1) % num_parties + 1}"
        key_rows.append([f"K{i + 1} ({sender}→{receiver})", key, key, "Match"])
    print(format_table(["Key", "Sender", "Receiver", "Status"], key_rows))

    print("\n7. Announcing differences...")
    diff_rows = []
    diff_lists = []
    for i in range(num_parties):
        incoming_idx = (i - 1) % num_parties
        comparison = f"K{incoming_idx + 1} (in) vs K{i + 1} (out)"
        diffs = find_different_positions(keys[incoming_idx], keys[i])
        diff_lists.append(diffs)
        diff_rows.append([f"P{i + 1}", comparison, str(diffs)])
    print(format_table(["Participant", "Comparison", "Different Positions"], diff_rows))

    print("\n8. Calculating final shared keys...")
    final_key = ''.join(keys)
    final_rows = [[f"P{i + 1}", final_key] for i in range(num_parties)]

    print("\n9. Final Shared Keys:")
    print(format_table(["Party", "Final Key"], final_rows))

    print("\nSuccess! All parties share the same final key")

    key = final_key
    key_length = len(key)
    num_zeros = key.count('0')
    num_ones = key.count('1')
    p0 = num_zeros / key_length if key_length else 0
    p1 = num_ones / key_length if key_length else 0
    entropy = 0
    for p in (p0, p1):
        if p > 0:
            entropy -= p * math.log2(p)
    latency = time.perf_counter() - t_start
    key_rate_bps = key_length / latency if latency > 0 else 0

    ghz_qubits = num_parties * M * 4
    decoy_qubits = num_parties * num_decoy
    quantum_qubits = ghz_qubits + decoy_qubits

    decoy_bits = num_parties * num_decoy * 2
    decoy_results_bits = num_parties * num_decoy
    diff_bits = sum(len(dl) for dl in diff_lists)
    classical_bits = decoy_bits + decoy_results_bits + diff_bits
    classical_messages = 2 * num_parties
    bit_flips = diff_bits

    print("\n==============================================")
    print("Final Key Statistical Analysis")
    print("==============================================")
    stats_rows = [
        ["Final key", key],
        ["Key length", f"{key_length} bits"],
        ["Number of 0s", num_zeros],
        ["Number of 1s", num_ones],
        ["Proportion of 0s", f"{p0:.2%}"],
        ["Proportion of 1s", f"{p1:.2%}"],
        ["Shannon entropy", f"{entropy:.4f} bits"],
        ["Key rate", f"{key_rate_bps:.4f} bit/s"],
        ["Latency", f"{latency:.4f} s"],
        ["Quantum resource cost", f"{quantum_qubits} qubits"],
        ["Classical resource cost", f"{classical_messages} messages, {classical_bits} classical bits"],
        ["Bit flips (key adjustment)", f"{bit_flips}"],
    ]
    print(format_table(["Statistic", "Value"], stats_rows))

    # Dynamic Adaptive Key Update Mechanism
    print("\n" + "=" * 60)
    print("Dynamic Adaptive Key Update Mechanism")
    print("=" * 60)

    scene = random.choice([1, 2, 3])
    KP_root = key

    final_key_used = KP_root
    scenario_name = "Initial QKA Only"
    final_participants = num_parties
    Q_val = 0
    W_val = 0

    if scene == 1:
        print("\n[Scene 1: Only Participants Withdrawal]")
        print("Scene 1 occurs, please select the number of leaving participants Q, where Q<n")
        Q = int(input("Enter number of leaving participants (Q): "))

        if Q >= num_parties:
            print(f"Error: Q ({Q}) must be less than n ({num_parties})")
            return

        remaining_parties = num_parties - Q
        print(f"\nParticipants leaving: {Q}")
        print(f"Remaining participants: {remaining_parties}")

        KP_new = scene1_key_update(remaining_parties, KP_root)
        Q_val = Q

        if KP_new:
            final_key_used = KP_new
            scenario_name = "Scene 1: Participants Withdrawal"
            final_participants = num_parties - Q_val
            print("\n" + "=" * 60)
            print("Key Update Summary")
            print("=" * 60)
            print(f"Old key (KP_Old): {KP_root}")
            print(f"New key (KP_New): {KP_new}")
            print(f"Key length: {len(KP_new)} bits")
            print("=" * 60)
        else:
            print("\nKey update failed!")

    elif scene == 2:
        print("\n[Scene 2: Only New Members Join]")
        print("Scene 2 occurs, please select the number of new participants W.")
        W = int(input("Enter number of new participants (W): "))

        if W <= 0:
            print("Error: W must be greater than 0")
            return

        print(f"\nNew participants joining: {W}")

        success, final_key, scene2_ghz_types = scene2_new_members_join(num_parties, W, KP_root, M, num_decoy, 'four')
        all_used_ghz_types.update(scene2_ghz_types)
        W_val = W

        if success:
            final_key_used = final_key
            scenario_name = "Scene 2: New Members Join"
            final_participants = num_parties + W_val
            print("\n" + "=" * 60)
            print("Key Distribution Summary")
            print("=" * 60)
            print(f"Root key (KP): {KP_root}")
            print(f"Final shared key: {final_key}")
            print(f"Total participants: {num_parties + W}")
            print(f"Key length: {len(final_key)} bits")
            print("=" * 60)
        else:
            print("\nKey distribution failed!")

    elif scene == 3:
        print("\n[Scene 3: Participants Withdrawal and New Members Join]")
        print(
            "Scene 3 occurs, please select the number of leaving participants Q, and please select the number of new participants W.")
        Q = int(input("Enter number of leaving participants (Q): "))
        W = int(input("Enter number of new participants (W): "))

        if Q >= num_parties:
            print(f"Error: Q ({Q}) must be less than n ({num_parties})")
            return

        if W <= 0:
            print("Error: W must be greater than 0")
            return

        print(f"\nParticipants leaving: {Q}")
        print(f"New participants joining: {W}")
        print(f"Remaining original participants: {num_parties - Q}")
        print(f"Total participants after update: {num_parties - Q + W}")

        KP_new, scene3_ghz_types = scene3_composite_scenario(num_parties, Q, W, KP_root, M, num_decoy, 'four')
        all_used_ghz_types.update(scene3_ghz_types)
        Q_val = Q
        W_val = W

        if KP_new:
            final_key_used = KP_new
            scenario_name = "Scene 3: Composite Scenario"
            final_participants = num_parties - Q_val + W_val
            print("\n" + "=" * 60)
            print("Composite Scenario Summary")
            print("=" * 60)
            print(f"Initial root key (KP): {KP_root}")
            print(f"Final new key (KP_New): {KP_new}")
            print(f"Key length: {len(KP_new)} bits")
            print("=" * 60)
        else:
            print("\nComposite scenario execution failed!")

    # Overall Statistical Analysis
    t_end = time.perf_counter()
    total_latency = t_end - t_start

    final_key_length = len(final_key_used) if final_key_used else 0
    num_zeros_final = final_key_used.count('0') if final_key_used else 0
    num_ones_final = final_key_used.count('1') if final_key_used else 0
    p0_final = num_zeros_final / final_key_length if final_key_length > 0 else 0
    p1_final = num_ones_final / final_key_length if final_key_length > 0 else 0
    entropy_final = 0
    for p in (p0_final, p1_final):
        if p > 0:
            entropy_final -= p * math.log2(p)

    initial_quantum_qubits = num_parties * M * 4 + num_parties * num_decoy
    initial_classical_cost = 2 * num_parties
    initial_decoy_bits = num_parties * num_decoy * 2
    initial_decoy_results_bits = num_parties * num_decoy
    initial_diff_bits = sum(len(d) for d in diff_lists)
    initial_classical_bits = initial_decoy_bits + initial_decoy_results_bits + initial_diff_bits

    additional_quantum_qubits = 0
    additional_classical_bits = 0
    additional_messages = 0

    if scene == 2:
        group_parties = 1 + W_val
        M_group = math.ceil(len(KP_root) / (group_parties * 2)) if len(KP_root) > 0 else M
        additional_quantum_qubits = group_parties * M_group * 4 + group_parties * num_decoy
        additional_classical_bits = group_parties * num_decoy * 2 + group_parties * num_decoy
        additional_messages = group_parties * 2
    elif scene == 3:
        group_parties = 1 + W_val
        M_group = math.ceil(len(KP_root) / (group_parties * 2)) if len(KP_root) > 0 else M
        additional_quantum_qubits = group_parties * M_group * 4 + group_parties * num_decoy
        additional_classical_bits = group_parties * num_decoy * 2 + group_parties * num_decoy
        additional_messages = group_parties * 2
        remaining_parties = num_parties - Q_val
        additional_classical_bits += remaining_parties * 32 * 2
        additional_messages += remaining_parties * 2
    elif scene == 1:
        remaining_parties = num_parties - Q_val
        additional_classical_bits = remaining_parties * 32 * 2
        additional_messages = remaining_parties * 2

    total_quantum_qubits = initial_quantum_qubits + additional_quantum_qubits
    total_classical_bits = initial_classical_bits + additional_classical_bits
    total_messages = initial_classical_cost + additional_messages

    overall_key_rate = final_key_length / total_latency if total_latency > 0 else 0
    ghz_types_formatted = ", ".join([f"G{i}" for i in sorted(all_used_ghz_types)])

    print("\n" + "=" * 60)
    print("Overall Program Statistical Analysis")
    print("=" * 60)
    overall_stats_rows = [
        ["Protocol scenario", scenario_name],
        ["Final key", final_key_used if final_key_used else "N/A"],
        ["Final key length", f"{final_key_length} bits"],
        ["Number of 0s", num_zeros_final],
        ["Number of 1s", num_ones_final],
        ["Proportion of 0s", f"{p0_final:.2%}"],
        ["Proportion of 1s", f"{p1_final:.2%}"],
        ["Shannon entropy", f"{entropy_final:.4f} bits"],
        ["Overall key rate", f"{overall_key_rate:.4f} bit/s"],
        ["Total latency", f"{total_latency:.4f} s"],
        ["Total quantum resource cost", f"{total_quantum_qubits} qubits"],
        ["  - Initial QKA", f"{initial_quantum_qubits} qubits"],
        ["  - Dynamic scenarios", f"{additional_quantum_qubits} qubits"],
        ["Total classical resource cost", f"{total_messages} messages, {total_classical_bits} classical bits"],
        ["  - Initial QKA", f"{initial_classical_cost} messages, {initial_classical_bits} classical bits"],
        ["  - Dynamic scenarios", f"{additional_messages} messages, {additional_classical_bits} classical bits"],
        ["Number of GHZ-like state types used", f"{len(all_used_ghz_types)} types: {ghz_types_formatted}"],
        ["Initial participants", f"{num_parties}"],
    ]

    if scene == 1:
        overall_stats_rows.append(["Participants leaving", f"{Q_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    elif scene == 2:
        overall_stats_rows.append(["Participants joining", f"{W_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    elif scene == 3:
        overall_stats_rows.append(["Participants leaving", f"{Q_val}"])
        overall_stats_rows.append(["Participants joining", f"{W_val}"])
        overall_stats_rows.append(["Final participants", f"{final_participants}"])
    else:
        overall_stats_rows.append(["Final participants", f"{final_participants}"])

    print(format_table(["Statistic", "Value"], overall_stats_rows))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Circuit Diagrams for All Used GHZ-like States")
    print("=" * 60)

    if all_used_ghz_types:
        sorted_ghz_types = sorted(all_used_ghz_types)
        ghz_types_list = [f"G{i}" for i in sorted_ghz_types]
        print(f"\nTotal GHZ-like state types used in this protocol execution: {len(all_used_ghz_types)}")
        print(f"Types: {', '.join(ghz_types_list)}")
        print(f"\nGenerating circuit diagrams for used GHZ-like states...")
        visualize_ghz_circuits(all_used_ghz_types, 'four')

        print("\n" + "=" * 60)
        print("Measurement Results for All Used GHZ-like States (1024 shots)")
        print("=" * 60)
        print(f"\nGenerating measurement results diagrams for used GHZ-like states...")
        visualize_ghz_measurement_results(all_used_ghz_types, 'four')
    else:
        print("\nNo GHZ-like states were used in this protocol execution.")
        print("(This may occur if only Scene 1 was executed, which uses hash-based key update without GHZ states)")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function - asks user to select particle type and runs the protocol"""
    print("=" * 60)
    print("N-party Dynamic Adaptive Multi-party Quantum Key Agreement (DA-MQKA)")
    print("=" * 60)
    print("\nPlease select the particle type for QKA protocol:")
    print("  1. Three-particle GHZ-like states (G0-G7)")
    print("  2. Four-particle GHZ-like states (G0-G15)")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    if choice == '1':
        particle_type = 'three'
        print("\nSelected: Three-particle GHZ-like states")
    elif choice == '2':
        particle_type = 'four'
        print("\nSelected: Four-particle GHZ-like states")
    else:
        print("Invalid choice. Defaulting to three-particle GHZ-like states.")
        particle_type = 'three'

    num_parties = int(input("\nEnter number of participants (n): "))
    M = int(input("Enter number of GHZ-like states per party (M): "))
    num_decoy = int(input("Enter number of decoy states to insert: "))

    if particle_type == 'three':
        run_n_party_protocol_three(M, num_decoy, num_parties)
    else:
        run_n_party_protocol_four(M, num_decoy, num_parties)


if __name__ == "__main__":
    main()

