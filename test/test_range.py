import pytest

from psst import Range, convert_to_range


class TestRange:
    @pytest.mark.parametrize("lo", [-2.5, -1e-5, 0, 1e-3, 2.0])
    def test_min_gt_max(self, lo: float):
        hi = lo - 1e-6
        with pytest.raises(ValueError):
            _ = Range(lo, hi, 1)

    @pytest.mark.parametrize("lo", [-2.5, -1e-5, -1e-30, 0])
    def test_neg_min_log_scale(self, lo: float):
        with pytest.raises(ValueError):
            _ = Range(lo, 2, 1, True)

    @pytest.mark.parametrize("shape", [8.4, 1.0, 2.0, (5, 2, 2, 1e3)])
    def test_float_in_shape(self, shape: float | tuple):
        with pytest.raises(TypeError):
            _ = Range(0, 1, shape)

    @pytest.mark.parametrize("shape", [0, -1, -2, (0, 2, 2), (1, 1, -1)])
    def test_nonnegative_in_shape(self, shape: int | tuple):
        with pytest.raises(ValueError):
            _ = Range(0, 1, shape)

    @pytest.mark.parametrize("shape", [1, 4, 8, 64, 1_000_321])
    def test_range_values_shape_int(self, shape: int):
        a = Range(0, 1, shape)
        assert a.shape == (shape,)

    @pytest.mark.parametrize("lo", [-2, 0, 1e-2])
    @pytest.mark.parametrize("hi", [0.2, 0.9, 10])
    @pytest.mark.parametrize("shape", [(1,), (8,), (2, 2), (8, 1, 1)])
    def test_range_values_linear(self, lo: float, hi: float, shape: tuple):
        a = Range(lo, hi, shape)
        assert a.min_value == lo
        assert a.max_value == hi
        assert a.shape == shape

    @pytest.mark.parametrize("lo", [1e-30, 9e-9, 2.71828e-2])
    @pytest.mark.parametrize("hi", [0.2, 0.9, 10])
    @pytest.mark.parametrize("shape", [(1,), (8,), (2, 2), (8, 1, 1)])
    def test_range_values_log(self, lo: float, hi: float, shape: tuple):
        a = Range(lo, hi, shape, log_scale=True)
        assert a.min_value == lo
        assert a.max_value == hi
        assert a.shape == shape


class TestConvertToRange:
    @pytest.mark.parametrize("x", [(0, 1, 1), (0, 1, (2, 1)), (0, 1, (1, 1, 8))])
    def test_tuple_to_range(self, x: tuple):
        a = convert_to_range(x)
        assert a.min_value == x[0]
        assert a.max_value == x[1]
        assert a.shape == x[2] or a.shape == (x[2],)

    @pytest.mark.parametrize("x", [(0, 1, 1), (0, 1, (2, 1)), (0, 1, (1, 1, 8))])
    def test_list_to_range(self, x: tuple):
        a = convert_to_range(list(x))
        assert a.min_value == x[0]
        assert a.max_value == x[1]
        assert a.shape == x[2] or a.shape == (x[2],)

    @pytest.mark.parametrize("x", [(0, 1, 1), (0, 1, (2, 1)), (0, 1, (1, 1, 8))])
    def test_dict_to_range(self, x: tuple):
        a = convert_to_range(dict(zip(["min_value", "max_value", "shape"], x)))
        assert a.min_value == x[0]
        assert a.max_value == x[1]
        assert a.shape == x[2] or a.shape == (x[2],)
