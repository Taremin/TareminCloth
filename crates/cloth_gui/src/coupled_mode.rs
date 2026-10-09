/// Coupled自己衝突モードの (値, ボタン名, ホバー説明)。
///
/// 値はコアの `dispatch.rs` の分岐と一致させる:
/// - 反復内で自己衝突を解くのは 2 と 3
/// - 外側で1回だけ解くのは 0 と 1
/// - 仕上げ直し (Post-Relaxation) を行うのは 1 と 3
///
/// Blender側の列挙 (`PER_ITERATION` / `FULL_COUPLED`) とは保存値の体系が異なる。
/// 変換は Python の `coupled_mode_from_settings` が担う。
pub const COUPLED_MODES: [(u32, &str, &str); 4] = [
    (0, "OFF", "外側で1回だけ解く。仕上げ直しなし"),
    (1, "Relaxation", "外側で解いた後に距離拘束を再適用する"),
    (2, "Per-Iteration", "ソルバー反復ごとに解く。仕上げ直しなし"),
    (3, "Full", "ソルバー反復ごとに解き、仕上げ直しも行う (Blender FULL_COUPLED と同じ)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_0_to_3_in_order() {
        let values: Vec<u32> = COUPLED_MODES.iter().map(|m| m.0).collect();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn full_label_maps_to_core_full() {
        // Blender の FULL_COUPLED が送る 3 と、GUIの Full ボタンが送る値を一致させる (R1の回帰防止)
        let full = COUPLED_MODES.iter().find(|m| m.1 == "Full").unwrap();
        assert_eq!(full.0, 3);
    }

    #[test]
    fn per_iteration_label_maps_to_core_mode_2() {
        let m = COUPLED_MODES.iter().find(|m| m.1 == "Per-Iteration").unwrap();
        assert_eq!(m.0, 2);
    }

    #[test]
    fn labels_are_unique_and_non_empty() {
        for (i, a) in COUPLED_MODES.iter().enumerate() {
            assert!(!a.1.is_empty() && !a.2.is_empty());
            for b in COUPLED_MODES.iter().skip(i + 1) {
                assert_ne!(a.1, b.1);
            }
        }
    }
}
