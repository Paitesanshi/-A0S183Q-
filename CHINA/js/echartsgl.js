! function(e, t) { "object" == typeof exports && "object" == typeof module ? module.exports = t(require("echarts")) : "function" == typeof define && define.amd ? define(["echarts"], t) : "object" == typeof exports ? exports["echarts-gl"] = t(require("echarts")) : e["echarts-gl"] = t(e.echarts) }(this, function(e) {
    return function(e) {
        function t(n) { if (r[n]) return r[n].exports; var i = r[n] = { i: n, l: !1, exports: {} }; return e[n].call(i.exports, i, i.exports, t), i.l = !0, i.exports }
        var r = {};
        return t.m = e, t.c = r, t.i = function(e) { return e }, t.d = function(e, r, n) { t.o(e, r) || Object.defineProperty(e, r, { configurable: !1, enumerable: !0, get: n }) }, t.n = function(e) { var r = e && e.__esModule ? function() { return e.default } : function() { return e }; return t.d(r, "a", r), r }, t.o = function(e, t) { return Object.prototype.hasOwnProperty.call(e, t) }, t.p = "", t(t.s = 99)
    }([function(t, r) { t.exports = e }, function(e, t, r) {
        ! function(e) {
            "use strict";
            var r = {};
            r.exports = t,
                function(e) {
                    if (!t) var t = 1e-6;
                    if (!r) var r = "undefined" != typeof Float32Array ? Float32Array : Array;
                    if (!n) var n = Math.random;
                    var i = {};
                    i.setMatrixArrayType = function(e) { r = e }, void 0 !== e && (e.glMatrix = i);
                    var a = Math.PI / 180;
                    i.toRadian = function(e) { return e * a };
                    var o = {};
                    o.create = function() { var e = new r(2); return e[0] = 0, e[1] = 0, e }, o.clone = function(e) { var t = new r(2); return t[0] = e[0], t[1] = e[1], t }, o.fromValues = function(e, t) { var n = new r(2); return n[0] = e, n[1] = t, n }, o.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e }, o.set = function(e, t, r) { return e[0] = t, e[1] = r, e }, o.add = function(e, t, r) { return e[0] = t[0] + r[0], e[1] = t[1] + r[1], e }, o.subtract = function(e, t, r) { return e[0] = t[0] - r[0], e[1] = t[1] - r[1], e }, o.sub = o.subtract, o.multiply = function(e, t, r) { return e[0] = t[0] * r[0], e[1] = t[1] * r[1], e }, o.mul = o.multiply, o.divide = function(e, t, r) { return e[0] = t[0] / r[0], e[1] = t[1] / r[1], e }, o.div = o.divide, o.min = function(e, t, r) { return e[0] = Math.min(t[0], r[0]), e[1] = Math.min(t[1], r[1]), e }, o.max = function(e, t, r) { return e[0] = Math.max(t[0], r[0]), e[1] = Math.max(t[1], r[1]), e }, o.scale = function(e, t, r) { return e[0] = t[0] * r, e[1] = t[1] * r, e }, o.scaleAndAdd = function(e, t, r, n) { return e[0] = t[0] + r[0] * n, e[1] = t[1] + r[1] * n, e }, o.distance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1];
                        return Math.sqrt(r * r + n * n)
                    }, o.dist = o.distance, o.squaredDistance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1];
                        return r * r + n * n
                    }, o.sqrDist = o.squaredDistance, o.length = function(e) {
                        var t = e[0],
                            r = e[1];
                        return Math.sqrt(t * t + r * r)
                    }, o.len = o.length, o.squaredLength = function(e) {
                        var t = e[0],
                            r = e[1];
                        return t * t + r * r
                    }, o.sqrLen = o.squaredLength, o.negate = function(e, t) { return e[0] = -t[0], e[1] = -t[1], e }, o.inverse = function(e, t) { return e[0] = 1 / t[0], e[1] = 1 / t[1], e }, o.normalize = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = r * r + n * n;
                        return i > 0 && (i = 1 / Math.sqrt(i), e[0] = t[0] * i, e[1] = t[1] * i), e
                    }, o.dot = function(e, t) { return e[0] * t[0] + e[1] * t[1] }, o.cross = function(e, t, r) { var n = t[0] * r[1] - t[1] * r[0]; return e[0] = e[1] = 0, e[2] = n, e }, o.lerp = function(e, t, r, n) {
                        var i = t[0],
                            a = t[1];
                        return e[0] = i + n * (r[0] - i), e[1] = a + n * (r[1] - a), e
                    }, o.random = function(e, t) { t = t || 1; var r = 2 * n() * Math.PI; return e[0] = Math.cos(r) * t, e[1] = Math.sin(r) * t, e }, o.transformMat2 = function(e, t, r) {
                        var n = t[0],
                            i = t[1];
                        return e[0] = r[0] * n + r[2] * i, e[1] = r[1] * n + r[3] * i, e
                    }, o.transformMat2d = function(e, t, r) {
                        var n = t[0],
                            i = t[1];
                        return e[0] = r[0] * n + r[2] * i + r[4], e[1] = r[1] * n + r[3] * i + r[5], e
                    }, o.transformMat3 = function(e, t, r) {
                        var n = t[0],
                            i = t[1];
                        return e[0] = r[0] * n + r[3] * i + r[6], e[1] = r[1] * n + r[4] * i + r[7], e
                    }, o.transformMat4 = function(e, t, r) {
                        var n = t[0],
                            i = t[1];
                        return e[0] = r[0] * n + r[4] * i + r[12], e[1] = r[1] * n + r[5] * i + r[13], e
                    }, o.forEach = function() { var e = o.create(); return function(t, r, n, i, a, o) { var s, u; for (r || (r = 2), n || (n = 0), u = i ? Math.min(i * r + n, t.length) : t.length, s = n; s < u; s += r) e[0] = t[s], e[1] = t[s + 1], a(e, e, o), t[s] = e[0], t[s + 1] = e[1]; return t } }(), o.str = function(e) { return "vec2(" + e[0] + ", " + e[1] + ")" }, void 0 !== e && (e.vec2 = o);
                    var s = {};
                    s.create = function() { var e = new r(3); return e[0] = 0, e[1] = 0, e[2] = 0, e }, s.clone = function(e) { var t = new r(3); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t }, s.fromValues = function(e, t, n) { var i = new r(3); return i[0] = e, i[1] = t, i[2] = n, i }, s.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e }, s.set = function(e, t, r, n) { return e[0] = t, e[1] = r, e[2] = n, e }, s.add = function(e, t, r) { return e[0] = t[0] + r[0], e[1] = t[1] + r[1], e[2] = t[2] + r[2], e }, s.subtract = function(e, t, r) { return e[0] = t[0] - r[0], e[1] = t[1] - r[1], e[2] = t[2] - r[2], e }, s.sub = s.subtract, s.multiply = function(e, t, r) { return e[0] = t[0] * r[0], e[1] = t[1] * r[1], e[2] = t[2] * r[2], e }, s.mul = s.multiply, s.divide = function(e, t, r) { return e[0] = t[0] / r[0], e[1] = t[1] / r[1], e[2] = t[2] / r[2], e }, s.div = s.divide, s.min = function(e, t, r) { return e[0] = Math.min(t[0], r[0]), e[1] = Math.min(t[1], r[1]), e[2] = Math.min(t[2], r[2]), e }, s.max = function(e, t, r) { return e[0] = Math.max(t[0], r[0]), e[1] = Math.max(t[1], r[1]), e[2] = Math.max(t[2], r[2]), e }, s.scale = function(e, t, r) { return e[0] = t[0] * r, e[1] = t[1] * r, e[2] = t[2] * r, e }, s.scaleAndAdd = function(e, t, r, n) { return e[0] = t[0] + r[0] * n, e[1] = t[1] + r[1] * n, e[2] = t[2] + r[2] * n, e }, s.distance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1],
                            i = t[2] - e[2];
                        return Math.sqrt(r * r + n * n + i * i)
                    }, s.dist = s.distance, s.squaredDistance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1],
                            i = t[2] - e[2];
                        return r * r + n * n + i * i
                    }, s.sqrDist = s.squaredDistance, s.length = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2];
                        return Math.sqrt(t * t + r * r + n * n)
                    }, s.len = s.length, s.squaredLength = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2];
                        return t * t + r * r + n * n
                    }, s.sqrLen = s.squaredLength, s.negate = function(e, t) { return e[0] = -t[0], e[1] = -t[1], e[2] = -t[2], e }, s.inverse = function(e, t) { return e[0] = 1 / t[0], e[1] = 1 / t[1], e[2] = 1 / t[2], e }, s.normalize = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = r * r + n * n + i * i;
                        return a > 0 && (a = 1 / Math.sqrt(a), e[0] = t[0] * a, e[1] = t[1] * a, e[2] = t[2] * a), e
                    }, s.dot = function(e, t) { return e[0] * t[0] + e[1] * t[1] + e[2] * t[2] }, s.cross = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = r[0],
                            s = r[1],
                            u = r[2];
                        return e[0] = i * u - a * s, e[1] = a * o - n * u, e[2] = n * s - i * o, e
                    }, s.lerp = function(e, t, r, n) {
                        var i = t[0],
                            a = t[1],
                            o = t[2];
                        return e[0] = i + n * (r[0] - i), e[1] = a + n * (r[1] - a), e[2] = o + n * (r[2] - o), e
                    }, s.random = function(e, t) {
                        t = t || 1;
                        var r = 2 * n() * Math.PI,
                            i = 2 * n() - 1,
                            a = Math.sqrt(1 - i * i) * t;
                        return e[0] = Math.cos(r) * a, e[1] = Math.sin(r) * a, e[2] = i * t, e
                    }, s.transformMat4 = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = r[3] * n + r[7] * i + r[11] * a + r[15];
                        return o = o || 1, e[0] = (r[0] * n + r[4] * i + r[8] * a + r[12]) / o, e[1] = (r[1] * n + r[5] * i + r[9] * a + r[13]) / o, e[2] = (r[2] * n + r[6] * i + r[10] * a + r[14]) / o, e
                    }, s.transformMat3 = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2];
                        return e[0] = n * r[0] + i * r[3] + a * r[6], e[1] = n * r[1] + i * r[4] + a * r[7], e[2] = n * r[2] + i * r[5] + a * r[8], e
                    }, s.transformQuat = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = r[0],
                            s = r[1],
                            u = r[2],
                            h = r[3],
                            l = h * n + s * a - u * i,
                            c = h * i + u * n - o * a,
                            d = h * a + o * i - s * n,
                            f = -o * n - s * i - u * a;
                        return e[0] = l * h + f * -o + c * -u - d * -s, e[1] = c * h + f * -s + d * -o - l * -u, e[2] = d * h + f * -u + l * -s - c * -o, e
                    }, s.rotateX = function(e, t, r, n) {
                        var i = [],
                            a = [];
                        return i[0] = t[0] - r[0], i[1] = t[1] - r[1], i[2] = t[2] - r[2], a[0] = i[0], a[1] = i[1] * Math.cos(n) - i[2] * Math.sin(n), a[2] = i[1] * Math.sin(n) + i[2] * Math.cos(n), e[0] = a[0] + r[0], e[1] = a[1] + r[1], e[2] = a[2] + r[2], e
                    }, s.rotateY = function(e, t, r, n) {
                        var i = [],
                            a = [];
                        return i[0] = t[0] - r[0], i[1] = t[1] - r[1], i[2] = t[2] - r[2], a[0] = i[2] * Math.sin(n) + i[0] * Math.cos(n), a[1] = i[1], a[2] = i[2] * Math.cos(n) - i[0] * Math.sin(n), e[0] = a[0] + r[0], e[1] = a[1] + r[1], e[2] = a[2] + r[2], e
                    }, s.rotateZ = function(e, t, r, n) {
                        var i = [],
                            a = [];
                        return i[0] = t[0] - r[0], i[1] = t[1] - r[1], i[2] = t[2] - r[2], a[0] = i[0] * Math.cos(n) - i[1] * Math.sin(n), a[1] = i[0] * Math.sin(n) + i[1] * Math.cos(n), a[2] = i[2], e[0] = a[0] + r[0], e[1] = a[1] + r[1], e[2] = a[2] + r[2], e
                    }, s.forEach = function() { var e = s.create(); return function(t, r, n, i, a, o) { var s, u; for (r || (r = 3), n || (n = 0), u = i ? Math.min(i * r + n, t.length) : t.length, s = n; s < u; s += r) e[0] = t[s], e[1] = t[s + 1], e[2] = t[s + 2], a(e, e, o), t[s] = e[0], t[s + 1] = e[1], t[s + 2] = e[2]; return t } }(), s.angle = function(e, t) {
                        var r = s.fromValues(e[0], e[1], e[2]),
                            n = s.fromValues(t[0], t[1], t[2]);
                        s.normalize(r, r), s.normalize(n, n);
                        var i = s.dot(r, n);
                        return i > 1 ? 0 : Math.acos(i)
                    }, s.str = function(e) { return "vec3(" + e[0] + ", " + e[1] + ", " + e[2] + ")" }, void 0 !== e && (e.vec3 = s);
                    var u = {};
                    u.create = function() { var e = new r(4); return e[0] = 0, e[1] = 0, e[2] = 0, e[3] = 0, e }, u.clone = function(e) { var t = new r(4); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t }, u.fromValues = function(e, t, n, i) { var a = new r(4); return a[0] = e, a[1] = t, a[2] = n, a[3] = i, a }, u.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e }, u.set = function(e, t, r, n, i) { return e[0] = t, e[1] = r, e[2] = n, e[3] = i, e }, u.add = function(e, t, r) { return e[0] = t[0] + r[0], e[1] = t[1] + r[1], e[2] = t[2] + r[2], e[3] = t[3] + r[3], e }, u.subtract = function(e, t, r) { return e[0] = t[0] - r[0], e[1] = t[1] - r[1], e[2] = t[2] - r[2], e[3] = t[3] - r[3], e }, u.sub = u.subtract, u.multiply = function(e, t, r) { return e[0] = t[0] * r[0], e[1] = t[1] * r[1], e[2] = t[2] * r[2], e[3] = t[3] * r[3], e }, u.mul = u.multiply, u.divide = function(e, t, r) { return e[0] = t[0] / r[0], e[1] = t[1] / r[1], e[2] = t[2] / r[2], e[3] = t[3] / r[3], e }, u.div = u.divide, u.min = function(e, t, r) { return e[0] = Math.min(t[0], r[0]), e[1] = Math.min(t[1], r[1]), e[2] = Math.min(t[2], r[2]), e[3] = Math.min(t[3], r[3]), e }, u.max = function(e, t, r) { return e[0] = Math.max(t[0], r[0]), e[1] = Math.max(t[1], r[1]), e[2] = Math.max(t[2], r[2]), e[3] = Math.max(t[3], r[3]), e }, u.scale = function(e, t, r) { return e[0] = t[0] * r, e[1] = t[1] * r, e[2] = t[2] * r, e[3] = t[3] * r, e }, u.scaleAndAdd = function(e, t, r, n) { return e[0] = t[0] + r[0] * n, e[1] = t[1] + r[1] * n, e[2] = t[2] + r[2] * n, e[3] = t[3] + r[3] * n, e }, u.distance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1],
                            i = t[2] - e[2],
                            a = t[3] - e[3];
                        return Math.sqrt(r * r + n * n + i * i + a * a)
                    }, u.dist = u.distance, u.squaredDistance = function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1],
                            i = t[2] - e[2],
                            a = t[3] - e[3];
                        return r * r + n * n + i * i + a * a
                    }, u.sqrDist = u.squaredDistance, u.length = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2],
                            i = e[3];
                        return Math.sqrt(t * t + r * r + n * n + i * i)
                    }, u.len = u.length, u.squaredLength = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2],
                            i = e[3];
                        return t * t + r * r + n * n + i * i
                    }, u.sqrLen = u.squaredLength, u.negate = function(e, t) { return e[0] = -t[0], e[1] = -t[1], e[2] = -t[2], e[3] = -t[3], e }, u.inverse = function(e, t) { return e[0] = 1 / t[0], e[1] = 1 / t[1], e[2] = 1 / t[2], e[3] = 1 / t[3], e }, u.normalize = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = r * r + n * n + i * i + a * a;
                        return o > 0 && (o = 1 / Math.sqrt(o), e[0] = t[0] * o, e[1] = t[1] * o, e[2] = t[2] * o, e[3] = t[3] * o), e
                    }, u.dot = function(e, t) { return e[0] * t[0] + e[1] * t[1] + e[2] * t[2] + e[3] * t[3] }, u.lerp = function(e, t, r, n) {
                        var i = t[0],
                            a = t[1],
                            o = t[2],
                            s = t[3];
                        return e[0] = i + n * (r[0] - i), e[1] = a + n * (r[1] - a), e[2] = o + n * (r[2] - o), e[3] = s + n * (r[3] - s), e
                    }, u.random = function(e, t) { return t = t || 1, e[0] = n(), e[1] = n(), e[2] = n(), e[3] = n(), u.normalize(e, e), u.scale(e, e, t), e }, u.transformMat4 = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3];
                        return e[0] = r[0] * n + r[4] * i + r[8] * a + r[12] * o, e[1] = r[1] * n + r[5] * i + r[9] * a + r[13] * o, e[2] = r[2] * n + r[6] * i + r[10] * a + r[14] * o, e[3] = r[3] * n + r[7] * i + r[11] * a + r[15] * o, e
                    }, u.transformQuat = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = r[0],
                            s = r[1],
                            u = r[2],
                            h = r[3],
                            l = h * n + s * a - u * i,
                            c = h * i + u * n - o * a,
                            d = h * a + o * i - s * n,
                            f = -o * n - s * i - u * a;
                        return e[0] = l * h + f * -o + c * -u - d * -s, e[1] = c * h + f * -s + d * -o - l * -u, e[2] = d * h + f * -u + l * -s - c * -o, e
                    }, u.forEach = function() { var e = u.create(); return function(t, r, n, i, a, o) { var s, u; for (r || (r = 4), n || (n = 0), u = i ? Math.min(i * r + n, t.length) : t.length, s = n; s < u; s += r) e[0] = t[s], e[1] = t[s + 1], e[2] = t[s + 2], e[3] = t[s + 3], a(e, e, o), t[s] = e[0], t[s + 1] = e[1], t[s + 2] = e[2], t[s + 3] = e[3]; return t } }(), u.str = function(e) { return "vec4(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ")" }, void 0 !== e && (e.vec4 = u);
                    var h = {};
                    h.create = function() { var e = new r(4); return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 1, e }, h.clone = function(e) { var t = new r(4); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t }, h.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e }, h.identity = function(e) { return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 1, e }, h.transpose = function(e, t) {
                        if (e === t) {
                            var r = t[1];
                            e[1] = t[2], e[2] = r
                        } else e[0] = t[0], e[1] = t[2], e[2] = t[1], e[3] = t[3];
                        return e
                    }, h.invert = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = r * a - i * n;
                        return o ? (o = 1 / o, e[0] = a * o, e[1] = -n * o, e[2] = -i * o, e[3] = r * o, e) : null
                    }, h.adjoint = function(e, t) { var r = t[0]; return e[0] = t[3], e[1] = -t[1], e[2] = -t[2], e[3] = r, e }, h.determinant = function(e) { return e[0] * e[3] - e[2] * e[1] }, h.multiply = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = r[0],
                            u = r[1],
                            h = r[2],
                            l = r[3];
                        return e[0] = n * s + a * u, e[1] = i * s + o * u, e[2] = n * h + a * l, e[3] = i * h + o * l, e
                    }, h.mul = h.multiply, h.rotate = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = Math.sin(r),
                            u = Math.cos(r);
                        return e[0] = n * u + a * s, e[1] = i * u + o * s, e[2] = n * -s + a * u, e[3] = i * -s + o * u, e
                    }, h.scale = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = r[0],
                            u = r[1];
                        return e[0] = n * s, e[1] = i * s, e[2] = a * u, e[3] = o * u, e
                    }, h.str = function(e) { return "mat2(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ")" }, h.frob = function(e) { return Math.sqrt(Math.pow(e[0], 2) + Math.pow(e[1], 2) + Math.pow(e[2], 2) + Math.pow(e[3], 2)) }, h.LDU = function(e, t, r, n) { return e[2] = n[2] / n[0], r[0] = n[0], r[1] = n[1], r[3] = n[3] - e[2] * r[1], [e, t, r] }, void 0 !== e && (e.mat2 = h);
                    var l = {};
                    l.create = function() { var e = new r(6); return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 1, e[4] = 0, e[5] = 0, e }, l.clone = function(e) { var t = new r(6); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t[4] = e[4], t[5] = e[5], t }, l.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e }, l.identity = function(e) { return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 1, e[4] = 0, e[5] = 0, e }, l.invert = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = r * a - n * i;
                        return u ? (u = 1 / u, e[0] = a * u, e[1] = -n * u, e[2] = -i * u, e[3] = r * u, e[4] = (i * s - a * o) * u, e[5] = (n * o - r * s) * u, e) : null
                    }, l.determinant = function(e) { return e[0] * e[3] - e[1] * e[2] }, l.multiply = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = r[0],
                            l = r[1],
                            c = r[2],
                            d = r[3],
                            f = r[4],
                            p = r[5];
                        return e[0] = n * h + a * l, e[1] = i * h + o * l, e[2] = n * c + a * d, e[3] = i * c + o * d, e[4] = n * f + a * p + s, e[5] = i * f + o * p + u, e
                    }, l.mul = l.multiply, l.rotate = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = Math.sin(r),
                            l = Math.cos(r);
                        return e[0] = n * l + a * h, e[1] = i * l + o * h, e[2] = n * -h + a * l, e[3] = i * -h + o * l, e[4] = s, e[5] = u, e
                    }, l.scale = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = r[0],
                            l = r[1];
                        return e[0] = n * h, e[1] = i * h, e[2] = a * l, e[3] = o * l, e[4] = s, e[5] = u, e
                    }, l.translate = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = r[0],
                            l = r[1];
                        return e[0] = n, e[1] = i, e[2] = a, e[3] = o, e[4] = n * h + a * l + s, e[5] = i * h + o * l + u, e
                    }, l.str = function(e) { return "mat2d(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ", " + e[4] + ", " + e[5] + ")" }, l.frob = function(e) { return Math.sqrt(Math.pow(e[0], 2) + Math.pow(e[1], 2) + Math.pow(e[2], 2) + Math.pow(e[3], 2) + Math.pow(e[4], 2) + Math.pow(e[5], 2) + 1) }, void 0 !== e && (e.mat2d = l);
                    var c = {};
                    c.create = function() { var e = new r(9); return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 1, e[5] = 0, e[6] = 0, e[7] = 0, e[8] = 1, e }, c.fromMat4 = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[4], e[4] = t[5], e[5] = t[6], e[6] = t[8], e[7] = t[9], e[8] = t[10], e }, c.clone = function(e) { var t = new r(9); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t[4] = e[4], t[5] = e[5], t[6] = e[6], t[7] = e[7], t[8] = e[8], t }, c.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e[6] = t[6], e[7] = t[7], e[8] = t[8], e }, c.identity = function(e) { return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 1, e[5] = 0, e[6] = 0, e[7] = 0, e[8] = 1, e }, c.transpose = function(e, t) {
                        if (e === t) {
                            var r = t[1],
                                n = t[2],
                                i = t[5];
                            e[1] = t[3], e[2] = t[6], e[3] = r, e[5] = t[7], e[6] = n, e[7] = i
                        } else e[0] = t[0], e[1] = t[3], e[2] = t[6], e[3] = t[1], e[4] = t[4], e[5] = t[7], e[6] = t[2], e[7] = t[5], e[8] = t[8];
                        return e
                    }, c.invert = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[7],
                            l = t[8],
                            c = l * o - s * h,
                            d = -l * a + s * u,
                            f = h * a - o * u,
                            p = r * c + n * d + i * f;
                        return p ? (p = 1 / p, e[0] = c * p, e[1] = (-l * n + i * h) * p, e[2] = (s * n - i * o) * p, e[3] = d * p, e[4] = (l * r - i * u) * p, e[5] = (-s * r + i * a) * p, e[6] = f * p, e[7] = (-h * r + n * u) * p, e[8] = (o * r - n * a) * p, e) : null
                    }, c.adjoint = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[7],
                            l = t[8];
                        return e[0] = o * l - s * h, e[1] = i * h - n * l, e[2] = n * s - i * o, e[3] = s * u - a * l, e[4] = r * l - i * u, e[5] = i * a - r * s, e[6] = a * h - o * u, e[7] = n * u - r * h, e[8] = r * o - n * a, e
                    }, c.determinant = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2],
                            i = e[3],
                            a = e[4],
                            o = e[5],
                            s = e[6],
                            u = e[7],
                            h = e[8];
                        return t * (h * a - o * u) + r * (-h * i + o * s) + n * (u * i - a * s)
                    }, c.multiply = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = t[6],
                            l = t[7],
                            c = t[8],
                            d = r[0],
                            f = r[1],
                            p = r[2],
                            _ = r[3],
                            m = r[4],
                            g = r[5],
                            v = r[6],
                            y = r[7],
                            x = r[8];
                        return e[0] = d * n + f * o + p * h, e[1] = d * i + f * s + p * l, e[2] = d * a + f * u + p * c, e[3] = _ * n + m * o + g * h, e[4] = _ * i + m * s + g * l, e[5] = _ * a + m * u + g * c, e[6] = v * n + y * o + x * h, e[7] = v * i + y * s + x * l, e[8] = v * a + y * u + x * c, e
                    }, c.mul = c.multiply, c.translate = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = t[6],
                            l = t[7],
                            c = t[8],
                            d = r[0],
                            f = r[1];
                        return e[0] = n, e[1] = i, e[2] = a, e[3] = o, e[4] = s, e[5] = u, e[6] = d * n + f * o + h, e[7] = d * i + f * s + l, e[8] = d * a + f * u + c, e
                    }, c.rotate = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = t[6],
                            l = t[7],
                            c = t[8],
                            d = Math.sin(r),
                            f = Math.cos(r);
                        return e[0] = f * n + d * o, e[1] = f * i + d * s, e[2] = f * a + d * u, e[3] = f * o - d * n, e[4] = f * s - d * i, e[5] = f * u - d * a, e[6] = h, e[7] = l, e[8] = c, e
                    }, c.scale = function(e, t, r) {
                        var n = r[0],
                            i = r[1];
                        return e[0] = n * t[0], e[1] = n * t[1], e[2] = n * t[2], e[3] = i * t[3], e[4] = i * t[4], e[5] = i * t[5], e[6] = t[6], e[7] = t[7], e[8] = t[8], e
                    }, c.fromMat2d = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = 0, e[3] = t[2], e[4] = t[3], e[5] = 0, e[6] = t[4], e[7] = t[5], e[8] = 1, e }, c.fromQuat = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = r + r,
                            s = n + n,
                            u = i + i,
                            h = r * o,
                            l = n * o,
                            c = n * s,
                            d = i * o,
                            f = i * s,
                            p = i * u,
                            _ = a * o,
                            m = a * s,
                            g = a * u;
                        return e[0] = 1 - c - p, e[3] = l - g, e[6] = d + m, e[1] = l + g, e[4] = 1 - h - p, e[7] = f - _, e[2] = d - m, e[5] = f + _, e[8] = 1 - h - c, e
                    }, c.normalFromMat4 = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[7],
                            l = t[8],
                            c = t[9],
                            d = t[10],
                            f = t[11],
                            p = t[12],
                            _ = t[13],
                            m = t[14],
                            g = t[15],
                            v = r * s - n * o,
                            y = r * u - i * o,
                            x = r * h - a * o,
                            T = n * u - i * s,
                            b = n * h - a * s,
                            w = i * h - a * u,
                            E = l * _ - c * p,
                            S = l * m - d * p,
                            A = l * g - f * p,
                            M = c * m - d * _,
                            N = c * g - f * _,
                            C = d * g - f * m,
                            L = v * C - y * N + x * M + T * A - b * S + w * E;
                        return L ? (L = 1 / L, e[0] = (s * C - u * N + h * M) * L, e[1] = (u * A - o * C - h * S) * L, e[2] = (o * N - s * A + h * E) * L, e[3] = (i * N - n * C - a * M) * L, e[4] = (r * C - i * A + a * S) * L, e[5] = (n * A - r * N - a * E) * L, e[6] = (_ * w - m * b + g * T) * L, e[7] = (m * x - p * w - g * y) * L, e[8] = (p * b - _ * x + g * v) * L, e) : null
                    }, c.str = function(e) { return "mat3(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ", " + e[4] + ", " + e[5] + ", " + e[6] + ", " + e[7] + ", " + e[8] + ")" }, c.frob = function(e) { return Math.sqrt(Math.pow(e[0], 2) + Math.pow(e[1], 2) + Math.pow(e[2], 2) + Math.pow(e[3], 2) + Math.pow(e[4], 2) + Math.pow(e[5], 2) + Math.pow(e[6], 2) + Math.pow(e[7], 2) + Math.pow(e[8], 2)) }, void 0 !== e && (e.mat3 = c);
                    var d = {};
                    d.create = function() { var e = new r(16); return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 1, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 1, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e }, d.clone = function(e) { var t = new r(16); return t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3], t[4] = e[4], t[5] = e[5], t[6] = e[6], t[7] = e[7], t[8] = e[8], t[9] = e[9], t[10] = e[10], t[11] = e[11], t[12] = e[12], t[13] = e[13], t[14] = e[14], t[15] = e[15], t }, d.copy = function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e[6] = t[6], e[7] = t[7], e[8] = t[8], e[9] = t[9], e[10] = t[10], e[11] = t[11], e[12] = t[12], e[13] = t[13], e[14] = t[14], e[15] = t[15], e }, d.identity = function(e) { return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 1, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 1, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e }, d.transpose = function(e, t) {
                        if (e === t) {
                            var r = t[1],
                                n = t[2],
                                i = t[3],
                                a = t[6],
                                o = t[7],
                                s = t[11];
                            e[1] = t[4], e[2] = t[8], e[3] = t[12], e[4] = r, e[6] = t[9], e[7] = t[13], e[8] = n, e[9] = a, e[11] = t[14], e[12] = i, e[13] = o, e[14] = s
                        } else e[0] = t[0], e[1] = t[4], e[2] = t[8], e[3] = t[12], e[4] = t[1], e[5] = t[5], e[6] = t[9], e[7] = t[13], e[8] = t[2], e[9] = t[6], e[10] = t[10], e[11] = t[14], e[12] = t[3], e[13] = t[7], e[14] = t[11], e[15] = t[15];
                        return e
                    }, d.invert = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[7],
                            l = t[8],
                            c = t[9],
                            d = t[10],
                            f = t[11],
                            p = t[12],
                            _ = t[13],
                            m = t[14],
                            g = t[15],
                            v = r * s - n * o,
                            y = r * u - i * o,
                            x = r * h - a * o,
                            T = n * u - i * s,
                            b = n * h - a * s,
                            w = i * h - a * u,
                            E = l * _ - c * p,
                            S = l * m - d * p,
                            A = l * g - f * p,
                            M = c * m - d * _,
                            N = c * g - f * _,
                            C = d * g - f * m,
                            L = v * C - y * N + x * M + T * A - b * S + w * E;
                        return L ? (L = 1 / L, e[0] = (s * C - u * N + h * M) * L, e[1] = (i * N - n * C - a * M) * L, e[2] = (_ * w - m * b + g * T) * L, e[3] = (d * b - c * w - f * T) * L, e[4] = (u * A - o * C - h * S) * L, e[5] = (r * C - i * A + a * S) * L, e[6] = (m * x - p * w - g * y) * L, e[7] = (l * w - d * x + f * y) * L, e[8] = (o * N - s * A + h * E) * L, e[9] = (n * A - r * N - a * E) * L, e[10] = (p * b - _ * x + g * v) * L, e[11] = (c * x - l * b - f * v) * L, e[12] = (s * S - o * M - u * E) * L, e[13] = (r * M - n * S + i * E) * L, e[14] = (_ * y - p * T - m * v) * L, e[15] = (l * T - c * y + d * v) * L, e) : null
                    }, d.adjoint = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[7],
                            l = t[8],
                            c = t[9],
                            d = t[10],
                            f = t[11],
                            p = t[12],
                            _ = t[13],
                            m = t[14],
                            g = t[15];
                        return e[0] = s * (d * g - f * m) - c * (u * g - h * m) + _ * (u * f - h * d), e[1] = -(n * (d * g - f * m) - c * (i * g - a * m) + _ * (i * f - a * d)), e[2] = n * (u * g - h * m) - s * (i * g - a * m) + _ * (i * h - a * u), e[3] = -(n * (u * f - h * d) - s * (i * f - a * d) + c * (i * h - a * u)), e[4] = -(o * (d * g - f * m) - l * (u * g - h * m) + p * (u * f - h * d)), e[5] = r * (d * g - f * m) - l * (i * g - a * m) + p * (i * f - a * d), e[6] = -(r * (u * g - h * m) - o * (i * g - a * m) + p * (i * h - a * u)), e[7] = r * (u * f - h * d) - o * (i * f - a * d) + l * (i * h - a * u), e[8] = o * (c * g - f * _) - l * (s * g - h * _) + p * (s * f - h * c), e[9] = -(r * (c * g - f * _) - l * (n * g - a * _) + p * (n * f - a * c)), e[10] = r * (s * g - h * _) - o * (n * g - a * _) + p * (n * h - a * s), e[11] = -(r * (s * f - h * c) - o * (n * f - a * c) + l * (n * h - a * s)), e[12] = -(o * (c * m - d * _) - l * (s * m - u * _) + p * (s * d - u * c)), e[13] = r * (c * m - d * _) - l * (n * m - i * _) + p * (n * d - i * c), e[14] = -(r * (s * m - u * _) - o * (n * m - i * _) + p * (n * u - i * s)), e[15] = r * (s * d - u * c) - o * (n * d - i * c) + l * (n * u - i * s), e
                    }, d.determinant = function(e) {
                        var t = e[0],
                            r = e[1],
                            n = e[2],
                            i = e[3],
                            a = e[4],
                            o = e[5],
                            s = e[6],
                            u = e[7],
                            h = e[8],
                            l = e[9],
                            c = e[10],
                            d = e[11],
                            f = e[12],
                            p = e[13],
                            _ = e[14],
                            m = e[15];
                        return (t * o - r * a) * (c * m - d * _) - (t * s - n * a) * (l * m - d * p) + (t * u - i * a) * (l * _ - c * p) + (r * s - n * o) * (h * m - d * f) - (r * u - i * o) * (h * _ - c * f) + (n * u - i * s) * (h * p - l * f)
                    }, d.multiply = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = t[4],
                            u = t[5],
                            h = t[6],
                            l = t[7],
                            c = t[8],
                            d = t[9],
                            f = t[10],
                            p = t[11],
                            _ = t[12],
                            m = t[13],
                            g = t[14],
                            v = t[15],
                            y = r[0],
                            x = r[1],
                            T = r[2],
                            b = r[3];
                        return e[0] = y * n + x * s + T * c + b * _, e[1] = y * i + x * u + T * d + b * m, e[2] = y * a + x * h + T * f + b * g, e[3] = y * o + x * l + T * p + b * v, y = r[4], x = r[5], T = r[6], b = r[7], e[4] = y * n + x * s + T * c + b * _, e[5] = y * i + x * u + T * d + b * m, e[6] = y * a + x * h + T * f + b * g, e[7] = y * o + x * l + T * p + b * v, y = r[8], x = r[9], T = r[10], b = r[11], e[8] = y * n + x * s + T * c + b * _, e[9] = y * i + x * u + T * d + b * m, e[10] = y * a + x * h + T * f + b * g, e[11] = y * o + x * l + T * p + b * v, y = r[12], x = r[13], T = r[14], b = r[15], e[12] = y * n + x * s + T * c + b * _, e[13] = y * i + x * u + T * d + b * m, e[14] = y * a + x * h + T * f + b * g, e[15] = y * o + x * l + T * p + b * v, e
                    }, d.multiplyAffine = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[4],
                            s = t[5],
                            u = t[6],
                            h = t[8],
                            l = t[9],
                            c = t[10],
                            d = t[12],
                            f = t[13],
                            p = t[14],
                            _ = r[0],
                            m = r[1],
                            g = r[2];
                        return e[0] = _ * n + m * o + g * h, e[1] = _ * i + m * s + g * l, e[2] = _ * a + m * u + g * c, _ = r[4], m = r[5], g = r[6], e[4] = _ * n + m * o + g * h, e[5] = _ * i + m * s + g * l, e[6] = _ * a + m * u + g * c, _ = r[8], m = r[9], g = r[10], e[8] = _ * n + m * o + g * h, e[9] = _ * i + m * s + g * l, e[10] = _ * a + m * u + g * c, _ = r[12], m = r[13], g = r[14], e[12] = _ * n + m * o + g * h + d, e[13] = _ * i + m * s + g * l + f, e[14] = _ * a + m * u + g * c + p, e
                    }, d.mul = d.multiply, d.mulAffine = d.multiplyAffine, d.translate = function(e, t, r) {
                        var n, i, a, o, s, u, h, l, c, d, f, p, _ = r[0],
                            m = r[1],
                            g = r[2];
                        return t === e ? (e[12] = t[0] * _ + t[4] * m + t[8] * g + t[12], e[13] = t[1] * _ + t[5] * m + t[9] * g + t[13], e[14] = t[2] * _ + t[6] * m + t[10] * g + t[14], e[15] = t[3] * _ + t[7] * m + t[11] * g + t[15]) : (n = t[0], i = t[1], a = t[2], o = t[3], s = t[4], u = t[5], h = t[6], l = t[7], c = t[8], d = t[9], f = t[10], p = t[11], e[0] = n, e[1] = i, e[2] = a, e[3] = o, e[4] = s, e[5] = u, e[6] = h, e[7] = l, e[8] = c, e[9] = d, e[10] = f, e[11] = p, e[12] = n * _ + s * m + c * g + t[12], e[13] = i * _ + u * m + d * g + t[13], e[14] = a * _ + h * m + f * g + t[14], e[15] = o * _ + l * m + p * g + t[15]), e
                    }, d.scale = function(e, t, r) {
                        var n = r[0],
                            i = r[1],
                            a = r[2];
                        return e[0] = t[0] * n, e[1] = t[1] * n, e[2] = t[2] * n, e[3] = t[3] * n, e[4] = t[4] * i, e[5] = t[5] * i, e[6] = t[6] * i, e[7] = t[7] * i, e[8] = t[8] * a, e[9] = t[9] * a, e[10] = t[10] * a, e[11] = t[11] * a, e[12] = t[12], e[13] = t[13], e[14] = t[14], e[15] = t[15], e
                    }, d.rotate = function(e, r, n, i) {
                        var a, o, s, u, h, l, c, d, f, p, _, m, g, v, y, x, T, b, w, E, S, A, M, N, C = i[0],
                            L = i[1],
                            D = i[2],
                            I = Math.sqrt(C * C + L * L + D * D);
                        return Math.abs(I) < t ? null : (I = 1 / I, C *= I, L *= I, D *= I, a = Math.sin(n), o = Math.cos(n), s = 1 - o, u = r[0], h = r[1], l = r[2], c = r[3], d = r[4], f = r[5], p = r[6], _ = r[7], m = r[8], g = r[9], v = r[10], y = r[11], x = C * C * s + o, T = L * C * s + D * a, b = D * C * s - L * a, w = C * L * s - D * a, E = L * L * s + o, S = D * L * s + C * a, A = C * D * s + L * a, M = L * D * s - C * a, N = D * D * s + o, e[0] = u * x + d * T + m * b, e[1] = h * x + f * T + g * b, e[2] = l * x + p * T + v * b, e[3] = c * x + _ * T + y * b, e[4] = u * w + d * E + m * S, e[5] = h * w + f * E + g * S, e[6] = l * w + p * E + v * S, e[7] = c * w + _ * E + y * S, e[8] = u * A + d * M + m * N, e[9] = h * A + f * M + g * N, e[10] = l * A + p * M + v * N, e[11] = c * A + _ * M + y * N, r !== e && (e[12] = r[12], e[13] = r[13], e[14] = r[14], e[15] = r[15]), e)
                    }, d.rotateX = function(e, t, r) {
                        var n = Math.sin(r),
                            i = Math.cos(r),
                            a = t[4],
                            o = t[5],
                            s = t[6],
                            u = t[7],
                            h = t[8],
                            l = t[9],
                            c = t[10],
                            d = t[11];
                        return t !== e && (e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[12] = t[12], e[13] = t[13], e[14] = t[14], e[15] = t[15]), e[4] = a * i + h * n, e[5] = o * i + l * n, e[6] = s * i + c * n, e[7] = u * i + d * n, e[8] = h * i - a * n, e[9] = l * i - o * n, e[10] = c * i - s * n, e[11] = d * i - u * n, e
                    }, d.rotateY = function(e, t, r) {
                        var n = Math.sin(r),
                            i = Math.cos(r),
                            a = t[0],
                            o = t[1],
                            s = t[2],
                            u = t[3],
                            h = t[8],
                            l = t[9],
                            c = t[10],
                            d = t[11];
                        return t !== e && (e[4] = t[4], e[5] = t[5], e[6] = t[6], e[7] = t[7], e[12] = t[12], e[13] = t[13], e[14] = t[14], e[15] = t[15]), e[0] = a * i - h * n, e[1] = o * i - l * n, e[2] = s * i - c * n, e[3] = u * i - d * n, e[8] = a * n + h * i, e[9] = o * n + l * i, e[10] = s * n + c * i, e[11] = u * n + d * i, e
                    }, d.rotateZ = function(e, t, r) {
                        var n = Math.sin(r),
                            i = Math.cos(r),
                            a = t[0],
                            o = t[1],
                            s = t[2],
                            u = t[3],
                            h = t[4],
                            l = t[5],
                            c = t[6],
                            d = t[7];
                        return t !== e && (e[8] = t[8], e[9] = t[9], e[10] = t[10], e[11] = t[11], e[12] = t[12], e[13] = t[13], e[14] = t[14], e[15] = t[15]), e[0] = a * i + h * n, e[1] = o * i + l * n, e[2] = s * i + c * n, e[3] = u * i + d * n, e[4] = h * i - a * n, e[5] = l * i - o * n, e[6] = c * i - s * n, e[7] = d * i - u * n, e
                    }, d.fromRotationTranslation = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = n + n,
                            u = i + i,
                            h = a + a,
                            l = n * s,
                            c = n * u,
                            d = n * h,
                            f = i * u,
                            p = i * h,
                            _ = a * h,
                            m = o * s,
                            g = o * u,
                            v = o * h;
                        return e[0] = 1 - (f + _), e[1] = c + v, e[2] = d - g, e[3] = 0, e[4] = c - v, e[5] = 1 - (l + _), e[6] = p + m, e[7] = 0, e[8] = d + g, e[9] = p - m, e[10] = 1 - (l + f), e[11] = 0, e[12] = r[0], e[13] = r[1], e[14] = r[2], e[15] = 1, e
                    }, d.fromQuat = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = r + r,
                            s = n + n,
                            u = i + i,
                            h = r * o,
                            l = n * o,
                            c = n * s,
                            d = i * o,
                            f = i * s,
                            p = i * u,
                            _ = a * o,
                            m = a * s,
                            g = a * u;
                        return e[0] = 1 - c - p, e[1] = l + g, e[2] = d - m, e[3] = 0, e[4] = l - g, e[5] = 1 - h - p, e[6] = f + _, e[7] = 0, e[8] = d + m, e[9] = f - _, e[10] = 1 - h - c, e[11] = 0, e[12] = 0, e[13] = 0, e[14] = 0, e[15] = 1, e
                    }, d.frustum = function(e, t, r, n, i, a, o) {
                        var s = 1 / (r - t),
                            u = 1 / (i - n),
                            h = 1 / (a - o);
                        return e[0] = 2 * a * s, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = 2 * a * u, e[6] = 0, e[7] = 0, e[8] = (r + t) * s, e[9] = (i + n) * u, e[10] = (o + a) * h, e[11] = -1, e[12] = 0, e[13] = 0, e[14] = o * a * 2 * h, e[15] = 0, e
                    }, d.perspective = function(e, t, r, n, i) {
                        var a = 1 / Math.tan(t / 2),
                            o = 1 / (n - i);
                        return e[0] = a / r, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = a, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = (i + n) * o, e[11] = -1, e[12] = 0, e[13] = 0, e[14] = 2 * i * n * o, e[15] = 0, e
                    }, d.ortho = function(e, t, r, n, i, a, o) {
                        var s = 1 / (t - r),
                            u = 1 / (n - i),
                            h = 1 / (a - o);
                        return e[0] = -2 * s, e[1] = 0, e[2] = 0, e[3] = 0, e[4] = 0, e[5] = -2 * u, e[6] = 0, e[7] = 0, e[8] = 0, e[9] = 0, e[10] = 2 * h, e[11] = 0, e[12] = (t + r) * s, e[13] = (i + n) * u, e[14] = (o + a) * h, e[15] = 1, e
                    }, d.lookAt = function(e, r, n, i) {
                        var a, o, s, u, h, l, c, f, p, _, m = r[0],
                            g = r[1],
                            v = r[2],
                            y = i[0],
                            x = i[1],
                            T = i[2],
                            b = n[0],
                            w = n[1],
                            E = n[2];
                        return Math.abs(m - b) < t && Math.abs(g - w) < t && Math.abs(v - E) < t ? d.identity(e) : (c = m - b, f = g - w, p = v - E, _ = 1 / Math.sqrt(c * c + f * f + p * p), c *= _, f *= _, p *= _, a = x * p - T * f, o = T * c - y * p, s = y * f - x * c, _ = Math.sqrt(a * a + o * o + s * s), _ ? (_ = 1 / _, a *= _, o *= _, s *= _) : (a = 0, o = 0, s = 0), u = f * s - p * o, h = p * a - c * s, l = c * o - f * a, _ = Math.sqrt(u * u + h * h + l * l), _ ? (_ = 1 / _, u *= _, h *= _, l *= _) : (u = 0, h = 0, l = 0), e[0] = a, e[1] = u, e[2] = c, e[3] = 0, e[4] = o, e[5] = h, e[6] = f, e[7] = 0, e[8] = s, e[9] = l, e[10] = p, e[11] = 0, e[12] = -(a * m + o * g + s * v), e[13] = -(u * m + h * g + l * v), e[14] = -(c * m + f * g + p * v), e[15] = 1, e)
                    }, d.str = function(e) { return "mat4(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ", " + e[4] + ", " + e[5] + ", " + e[6] + ", " + e[7] + ", " + e[8] + ", " + e[9] + ", " + e[10] + ", " + e[11] + ", " + e[12] + ", " + e[13] + ", " + e[14] + ", " + e[15] + ")" }, d.frob = function(e) { return Math.sqrt(Math.pow(e[0], 2) + Math.pow(e[1], 2) + Math.pow(e[2], 2) + Math.pow(e[3], 2) + Math.pow(e[4], 2) + Math.pow(e[5], 2) + Math.pow(e[6], 2) + Math.pow(e[7], 2) + Math.pow(e[8], 2) + Math.pow(e[9], 2) + Math.pow(e[10], 2) + Math.pow(e[11], 2) + Math.pow(e[12], 2) + Math.pow(e[13], 2) + Math.pow(e[14], 2) + Math.pow(e[15], 2)) }, void 0 !== e && (e.mat4 = d);
                    var f = {};
                    f.create = function() { var e = new r(4); return e[0] = 0, e[1] = 0, e[2] = 0, e[3] = 1, e }, f.rotationTo = function() {
                        var e = s.create(),
                            t = s.fromValues(1, 0, 0),
                            r = s.fromValues(0, 1, 0);
                        return function(n, i, a) { var o = s.dot(i, a); return o < -.999999 ? (s.cross(e, t, i), s.length(e) < 1e-6 && s.cross(e, r, i), s.normalize(e, e), f.setAxisAngle(n, e, Math.PI), n) : o > .999999 ? (n[0] = 0, n[1] = 0, n[2] = 0, n[3] = 1, n) : (s.cross(e, i, a), n[0] = e[0], n[1] = e[1], n[2] = e[2], n[3] = 1 + o, f.normalize(n, n)) }
                    }(), f.setAxes = function() { var e = c.create(); return function(t, r, n, i) { return e[0] = n[0], e[3] = n[1], e[6] = n[2], e[1] = i[0], e[4] = i[1], e[7] = i[2], e[2] = -r[0], e[5] = -r[1], e[8] = -r[2], f.normalize(t, f.fromMat3(t, e)) } }(), f.clone = u.clone, f.fromValues = u.fromValues, f.copy = u.copy, f.set = u.set, f.identity = function(e) { return e[0] = 0, e[1] = 0, e[2] = 0, e[3] = 1, e }, f.setAxisAngle = function(e, t, r) { r *= .5; var n = Math.sin(r); return e[0] = n * t[0], e[1] = n * t[1], e[2] = n * t[2], e[3] = Math.cos(r), e }, f.add = u.add, f.multiply = function(e, t, r) {
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = r[0],
                            u = r[1],
                            h = r[2],
                            l = r[3];
                        return e[0] = n * l + o * s + i * h - a * u, e[1] = i * l + o * u + a * s - n * h, e[2] = a * l + o * h + n * u - i * s, e[3] = o * l - n * s - i * u - a * h, e
                    }, f.mul = f.multiply, f.scale = u.scale, f.rotateX = function(e, t, r) {
                        r *= .5;
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = Math.sin(r),
                            u = Math.cos(r);
                        return e[0] = n * u + o * s, e[1] = i * u + a * s, e[2] = a * u - i * s, e[3] = o * u - n * s, e
                    }, f.rotateY = function(e, t, r) {
                        r *= .5;
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = Math.sin(r),
                            u = Math.cos(r);
                        return e[0] = n * u - a * s, e[1] = i * u + o * s, e[2] = a * u + n * s, e[3] = o * u - i * s, e
                    }, f.rotateZ = function(e, t, r) {
                        r *= .5;
                        var n = t[0],
                            i = t[1],
                            a = t[2],
                            o = t[3],
                            s = Math.sin(r),
                            u = Math.cos(r);
                        return e[0] = n * u + i * s, e[1] = i * u - n * s, e[2] = a * u + o * s, e[3] = o * u - a * s, e
                    }, f.calculateW = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2];
                        return e[0] = r, e[1] = n, e[2] = i, e[3] = Math.sqrt(Math.abs(1 - r * r - n * n - i * i)), e
                    }, f.dot = u.dot, f.lerp = u.lerp, f.slerp = function(e, t, r, n) {
                        var i, a, o, s, u, h = t[0],
                            l = t[1],
                            c = t[2],
                            d = t[3],
                            f = r[0],
                            p = r[1],
                            _ = r[2],
                            m = r[3];
                        return a = h * f + l * p + c * _ + d * m, a < 0 && (a = -a, f = -f, p = -p, _ = -_, m = -m), 1 - a > 1e-6 ? (i = Math.acos(a), o = Math.sin(i), s = Math.sin((1 - n) * i) / o, u = Math.sin(n * i) / o) : (s = 1 - n, u = n), e[0] = s * h + u * f, e[1] = s * l + u * p, e[2] = s * c + u * _, e[3] = s * d + u * m, e
                    }, f.invert = function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = t[2],
                            a = t[3],
                            o = r * r + n * n + i * i + a * a,
                            s = o ? 1 / o : 0;
                        return e[0] = -r * s, e[1] = -n * s, e[2] = -i * s, e[3] = a * s, e
                    }, f.conjugate = function(e, t) { return e[0] = -t[0], e[1] = -t[1], e[2] = -t[2], e[3] = t[3], e }, f.length = u.length, f.len = f.length, f.squaredLength = u.squaredLength, f.sqrLen = f.squaredLength, f.normalize = u.normalize, f.fromMat3 = function(e, t) {
                        var r, n = t[0] + t[4] + t[8];
                        if (n > 0) r = Math.sqrt(n + 1), e[3] = .5 * r, r = .5 / r, e[0] = (t[5] - t[7]) * r, e[1] = (t[6] - t[2]) * r, e[2] = (t[1] - t[3]) * r;
                        else {
                            var i = 0;
                            t[4] > t[0] && (i = 1), t[8] > t[3 * i + i] && (i = 2);
                            var a = (i + 1) % 3,
                                o = (i + 2) % 3;
                            r = Math.sqrt(t[3 * i + i] - t[3 * a + a] - t[3 * o + o] + 1), e[i] = .5 * r, r = .5 / r, e[3] = (t[3 * a + o] - t[3 * o + a]) * r, e[a] = (t[3 * a + i] + t[3 * i + a]) * r, e[o] = (t[3 * o + i] + t[3 * i + o]) * r
                        }
                        return e
                    }, f.str = function(e) { return "quat(" + e[0] + ", " + e[1] + ", " + e[2] + ", " + e[3] + ")" }, void 0 !== e && (e.quat = f)
                }(r.exports)
        }()
    }, function(e, t, r) {
        function n(e) { return !e || "none" === e }

        function i(e) { return e instanceof HTMLCanvasElement || e instanceof HTMLImageElement || e instanceof Image }

        function a(e) { return e.getZr && e.setOption }

        function o(e) { return Math.pow(2, Math.round(Math.log(e) / Math.LN2)) }

        function s(e) {
            if ((e.wrapS === c.REPEAT || e.wrapT === c.REPEAT) && e.image) {
                var t = o(e.width),
                    r = o(e.height);
                if (t !== e.width || r !== e.height) {
                    var n = document.createElement("canvas");
                    n.width = t, n.height = r;
                    n.getContext("2d").drawImage(e.image, 0, 0, t, r), e.image = n
                }
            }
        }
        var u = r(25),
            h = r(52),
            l = r(5),
            c = r(6),
            d = r(7),
            f = r(16),
            p = r(35),
            _ = r(13),
            m = r(0),
            g = r(26),
            v = r(60),
            y = r(47),
            x = r(169),
            T = r(203),
            b = r(204),
            w = r(231),
            E = r(4),
            S = r(171);
        m.util.extend(p.prototype, S), d.import(r(227)), d.import(r(82)), d.import(r(179)), d.import(r(178)), d.import(r(183)), d.import(r(186)), d.import(r(181)), d.import(r(187));
        var A = g.prototype.addToScene,
            M = g.prototype.removeFromScene;
        g.prototype.addToScene = function(e) {
            if (A.call(this, e), this.__zr) {
                var t = this.__zr;
                e.traverse(function(e) { e.__zr = t, e.addAnimatorsToZr && e.addAnimatorsToZr(t) })
            }
        }, g.prototype.removeFromScene = function(e) {
            M.call(this, e), e.traverse(function(e) {
                var t = e.__zr;
                e.__zr = null, t && e.removeAnimatorsFromZr && e.removeAnimatorsFromZr(t)
            })
        }, f.prototype.setTextureImage = function(e, t, r, i) {
            if (this.shader) {
                var a, o = r.getZr(),
                    s = this;
                return s.shader.disableTexture(e), n(t) || (a = N.loadTexture(t, r, i, function(t) { s.shader.enableTexture(e), o && o.refresh() }), s.set(e, a)), a
            }
        };
        var N = {};
        N.Renderer = h, N.Node = p, N.Mesh = u, N.Shader = d, N.Material = f, N.Texture = c, N.Texture2D = l, N.Geometry = _, N.SphereGeometry = r(75), N.PlaneGeometry = r(46), N.CubeGeometry = r(74), N.AmbientLight = r(202), N.DirectionalLight = r(76), N.PointLight = r(77), N.SpotLight = r(78), N.PerspectiveCamera = r(44), N.OrthographicCamera = r(36), N.Vector2 = r(28), N.Vector3 = r(3), N.Vector4 = r(209), N.Quaternion = r(55), N.Matrix2 = r(206), N.Matrix2d = r(207), N.Matrix3 = r(208), N.Matrix4 = r(9), N.Plane = r(79), N.Ray = r(56), N.BoundingBox = r(14), N.Frustum = r(54);
        var C = y.createBlank("rgba(255,255,255,0)").image;
        N.loadTexture = function(e, t, r, n) {
            "function" == typeof r && (n = r, r = {}), r = r || {};
            for (var o = Object.keys(r).sort(), u = "", h = 0; h < o.length; h++) u += o[h] + "_" + r[o[h]] + "_";
            var l = t.__textureCache = t.__textureCache || new v(20);
            if (a(e)) {
                var c = e.__textureid__,
                    d = l.get(u + c);
                if (d) d.texture.surface.setECharts(e), n && n(d.texture);
                else {
                    var f = new x(e);
                    f.onupdate = function() { t.getZr().refresh() }, d = { texture: f.getTexture() };
                    for (var h = 0; h < o.length; h++) d.texture[o[h]] = r[o[h]];
                    c = e.__textureid__ || "__ecgl_ec__" + d.texture.__GUID__, e.__textureid__ = c, l.put(u + c, d), n && n(d.texture)
                }
                return d.texture
            }
            if (i(e)) {
                var c = e.__textureid__,
                    d = l.get(u + c);
                if (!d) {
                    d = { texture: new N.Texture2D({ image: e }) };
                    for (var h = 0; h < o.length; h++) d.texture[o[h]] = r[o[h]];
                    c = e.__textureid__ || "__ecgl_image__" + d.texture.__GUID__, e.__textureid__ = c, l.put(u + c, d), s(d.texture), n && n(d.texture)
                }
                return d.texture
            }
            var d = l.get(u + e);
            if (d) d.callbacks ? d.callbacks.push(n) : n && n(d.texture);
            else if (e.match(/.hdr$|^data:application\/octet-stream/)) {
                d = { callbacks: [n] };
                var p = y.loadTexture(e, { exposure: r.exposure, fileType: "hdr" }, function() { p.dirty(), d.callbacks.forEach(function(e) { e && e(p) }), d.callbacks = null });
                d.texture = p, l.put(u + e, d)
            } else {
                for (var p = new N.Texture2D({ image: new Image }), h = 0; h < o.length; h++) p[o[h]] = r[o[h]];
                d = { texture: p, callbacks: [n] };
                var _ = p.image;
                _.onload = function() { p.image = _, s(p), p.dirty(), d.callbacks.forEach(function(e) { e && e(p) }), d.callbacks = null }, _.src = e, p.image = C, l.put(u + e, d)
            }
            return d.texture
        }, N.createAmbientCubemap = function(e, t, r, n) {
            e = e || {};
            var i = e.texture,
                a = E.firstNotNull(e.exposure, 1),
                o = new T({ intensity: E.firstNotNull(e.specularIntensity, 1) }),
                s = new b({ intensity: E.firstNotNull(e.diffuseIntensity, 1), coefficients: [.844, .712, .691, -.037, .083, .167, .343, .288, .299, -.041, -.021, -.009, -.003, -.041, -.064, -.011, -.007, -.004, -.031, .034, .081, -.06, -.049, -.06, .046, .056, .05] });
            return o.cubemap = N.loadTexture(i, r, { exposure: a }, function() { o.cubemap.flipY = !1, o.prefilter(t, 32), s.coefficients = w.projectEnvironmentMap(t, o.cubemap, { lod: 1 }), n && n() }), { specular: o, diffuse: s }
        }, N.createBlankTexture = y.createBlank, N.isImage = i, N.additiveBlend = function(e) { e.blendEquation(e.FUNC_ADD), e.blendFunc(e.SRC_ALPHA, e.ONE) }, N.parseColor = function(e, t) { return e instanceof Array ? (t || (t = []), t[0] = e[0], t[1] = e[1], t[2] = e[2], e.length > 3 ? t[3] = e[3] : t[3] = 1, t) : (t = m.color.parse(e || "#000", t) || [0, 0, 0, 0], t[0] /= 255, t[1] /= 255, t[2] /= 255, t) }, N.directionFromAlphaBeta = function(e, t) {
            var r = e / 180 * Math.PI + Math.PI / 2,
                n = -t / 180 * Math.PI + Math.PI / 2,
                i = [],
                a = Math.sin(r);
            return i[0] = a * Math.cos(n), i[1] = -Math.cos(r), i[2] = a * Math.sin(n), i
        }, N.getShadowResolution = function(e) {
            var t = 1024;
            switch (e) {
                case "low":
                    t = 512;
                    break;
                case "medium":
                    break;
                case "high":
                    t = 2048;
                    break;
                case "ultra":
                    t = 4096
            }
            return t
        }, N.COMMON_SHADERS = ["lambert", "color", "realistic", "hatching"], N.createShader = function(e) {
            var t = d.source(e + ".vertex"),
                r = d.source(e + ".fragment");
            return t || console.error("Vertex shader of '%s' not exits", e), r || console.error("Fragment shader of '%s' not exits", e), new d({ name: e, vertex: t, fragment: r })
        }, N.setMaterialFromModel = function(e, t, r, n) {
            var i = r.getModel(e + "Material"),
                a = i.get("detailTexture"),
                o = E.firstNotNull(i.get("textureTiling"), 1),
                s = E.firstNotNull(i.get("textureOffset"), 0);
            "number" == typeof o && (o = [o, o]), "number" == typeof s && (s = [s, s]);
            var u = o[0] > 1 || o[1] > 1 ? N.Texture.REPEAT : N.Texture.CLAMP_TO_EDGE,
                h = { anisotropic: 8, wrapS: u, wrapT: u };
            if ("realistic" === e) {
                var l = i.get("roughness"),
                    c = i.get("metalness");
                null != c ? isNaN(c) && (t.setTextureImage("metalnessMap", c, n, h), c = E.firstNotNull(i.get("metalnessAdjust"), .5)) : c = 0, null != l ? isNaN(l) && (t.setTextureImage("roughnessMap", l, n, h), l = E.firstNotNull(i.get("roughnessAdjust"), .5)) : l = .5;
                var d = i.get("normalTexture");
                t.setTextureImage("detailMap", a, n, h), t.setTextureImage("normalMap", d, n, h), t.set({ roughness: l, metalness: c, detailUvRepeat: o, detailUvOffset: s })
            } else if ("lambert" === e) t.setTextureImage("detailMap", a, n, h), t.set({ detailUvRepeat: o, detailUvOffset: s });
            else if ("color" === e) t.setTextureImage("detailMap", a, n, h), t.set({ detailUvRepeat: o, detailUvOffset: s });
            else if ("hatching" === e) {
                for (var f = i.get("hatchingTextures") || [], p = 0; p < 6; p++) t.setTextureImage("hatch" + (p + 1), f[p], n, { anisotropic: 8, wrapS: N.Texture.REPEAT, wrapT: N.Texture.REPEAT });
                t.set({ detailUvRepeat: o, detailUvOffset: s })
            }
        }, N.updateVertexAnimation = function(e, t, r, n) {
            var i = n.get("animation"),
                a = n.get("animationDurationUpdate"),
                o = n.get("animationEasingUpdate"),
                s = r.shadowDepthMaterial;
            if (i && t && a > 0 && t.geometry.vertexCount === r.geometry.vertexCount) {
                r.material.shader.define("vertex", "VERTEX_ANIMATION"), r.ignorePreZ = !0, s && s.shader.define("vertex", "VERTEX_ANIMATION");
                for (var u = 0; u < e.length; u++) r.geometry.attributes[e[u][0]].value = t.geometry.attributes[e[u][1]].value;
                r.geometry.dirty(), r.__percent = 0, r.material.set("percent", 0), r.stopAnimation(), r.animate().when(a, { __percent: 1 }).during(function() { r.material.set("percent", r.__percent), s && s.set("percent", r.__percent) }).done(function() { r.ignorePreZ = !1, r.material.shader.undefine("vertex", "VERTEX_ANIMATION"), s && s.shader.undefine("vertex", "VERTEX_ANIMATION") }).start(o)
            } else r.material.shader.undefine("vertex", "VERTEX_ANIMATION"), s && s.shader.undefine("vertex", "VERTEX_ANIMATION")
        }, e.exports = N
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r) { return e < t ? t : e > r ? r : e }
        var i = r(1),
            a = i.vec3,
            o = function(e, t, r) { e = e || 0, t = t || 0, r = r || 0, this._array = a.fromValues(e, t, r), this._dirty = !0 };
        o.prototype = {
            constructor: o,
            add: function(e) { return a.add(this._array, this._array, e._array), this._dirty = !0, this },
            set: function(e, t, r) { return this._array[0] = e, this._array[1] = t, this._array[2] = r, this._dirty = !0, this },
            setArray: function(e) { return this._array[0] = e[0], this._array[1] = e[1], this._array[2] = e[2], this._dirty = !0, this },
            clone: function() { return new o(this.x, this.y, this.z) },
            copy: function(e) { return a.copy(this._array, e._array), this._dirty = !0, this },
            cross: function(e, t) { return a.cross(this._array, e._array, t._array), this._dirty = !0, this },
            dist: function(e) { return a.dist(this._array, e._array) },
            distance: function(e) { return a.distance(this._array, e._array) },
            div: function(e) { return a.div(this._array, this._array, e._array), this._dirty = !0, this },
            divide: function(e) { return a.divide(this._array, this._array, e._array), this._dirty = !0, this },
            dot: function(e) { return a.dot(this._array, e._array) },
            len: function() { return a.len(this._array) },
            length: function() { return a.length(this._array) },
            lerp: function(e, t, r) { return a.lerp(this._array, e._array, t._array, r), this._dirty = !0, this },
            min: function(e) { return a.min(this._array, this._array, e._array), this._dirty = !0, this },
            max: function(e) { return a.max(this._array, this._array, e._array), this._dirty = !0, this },
            mul: function(e) { return a.mul(this._array, this._array, e._array), this._dirty = !0, this },
            multiply: function(e) { return a.multiply(this._array, this._array, e._array), this._dirty = !0, this },
            negate: function() { return a.negate(this._array, this._array), this._dirty = !0, this },
            normalize: function() { return a.normalize(this._array, this._array), this._dirty = !0, this },
            random: function(e) { return a.random(this._array, e), this._dirty = !0, this },
            scale: function(e) { return a.scale(this._array, this._array, e), this._dirty = !0, this },
            scaleAndAdd: function(e, t) { return a.scaleAndAdd(this._array, this._array, e._array, t), this._dirty = !0, this },
            sqrDist: function(e) { return a.sqrDist(this._array, e._array) },
            squaredDistance: function(e) { return a.squaredDistance(this._array, e._array) },
            sqrLen: function() { return a.sqrLen(this._array) },
            squaredLength: function() { return a.squaredLength(this._array) },
            sub: function(e) { return a.sub(this._array, this._array, e._array), this._dirty = !0, this },
            subtract: function(e) { return a.subtract(this._array, this._array, e._array), this._dirty = !0, this },
            transformMat3: function(e) { return a.transformMat3(this._array, this._array, e._array), this._dirty = !0, this },
            transformMat4: function(e) { return a.transformMat4(this._array, this._array, e._array), this._dirty = !0, this },
            transformQuat: function(e) { return a.transformQuat(this._array, this._array, e._array), this._dirty = !0, this },
            applyProjection: function(e) {
                var t = this._array;
                if (e = e._array, 0 === e[15]) {
                    var r = -1 / t[2];
                    t[0] = e[0] * t[0] * r, t[1] = e[5] * t[1] * r, t[2] = (e[10] * t[2] + e[14]) * r
                } else t[0] = e[0] * t[0] + e[12], t[1] = e[5] * t[1] + e[13], t[2] = e[10] * t[2] + e[14];
                return this._dirty = !0, this
            },
            eulerFromQuat: function(e, t) { o.eulerFromQuat(this, e, t) },
            eulerFromMat3: function(e, t) { o.eulerFromMat3(this, e, t) },
            toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" },
            toArray: function() { return Array.prototype.slice.call(this._array) }
        };
        var s = Object.defineProperty;
        if (s) {
            var u = o.prototype;
            s(u, "x", { get: function() { return this._array[0] }, set: function(e) { this._array[0] = e, this._dirty = !0 } }), s(u, "y", { get: function() { return this._array[1] }, set: function(e) { this._array[1] = e, this._dirty = !0 } }), s(u, "z", { get: function() { return this._array[2] }, set: function(e) { this._array[2] = e, this._dirty = !0 } })
        }
        o.add = function(e, t, r) { return a.add(e._array, t._array, r._array), e._dirty = !0, e }, o.set = function(e, t, r, n) { a.set(e._array, t, r, n), e._dirty = !0 }, o.copy = function(e, t) { return a.copy(e._array, t._array), e._dirty = !0, e }, o.cross = function(e, t, r) { return a.cross(e._array, t._array, r._array), e._dirty = !0, e }, o.dist = function(e, t) { return a.distance(e._array, t._array) }, o.distance = o.dist, o.div = function(e, t, r) { return a.divide(e._array, t._array, r._array), e._dirty = !0, e }, o.divide = o.div, o.dot = function(e, t) { return a.dot(e._array, t._array) }, o.len = function(e) { return a.length(e._array) }, o.lerp = function(e, t, r, n) { return a.lerp(e._array, t._array, r._array, n), e._dirty = !0, e }, o.min = function(e, t, r) { return a.min(e._array, t._array, r._array), e._dirty = !0, e }, o.max = function(e, t, r) { return a.max(e._array, t._array, r._array), e._dirty = !0, e }, o.mul = function(e, t, r) { return a.multiply(e._array, t._array, r._array), e._dirty = !0, e }, o.multiply = o.mul, o.negate = function(e, t) { return a.negate(e._array, t._array), e._dirty = !0, e }, o.normalize = function(e, t) { return a.normalize(e._array, t._array), e._dirty = !0, e }, o.random = function(e, t) { return a.random(e._array, t), e._dirty = !0, e }, o.scale = function(e, t, r) { return a.scale(e._array, t._array, r), e._dirty = !0, e }, o.scaleAndAdd = function(e, t, r, n) { return a.scaleAndAdd(e._array, t._array, r._array, n), e._dirty = !0, e }, o.sqrDist = function(e, t) { return a.sqrDist(e._array, t._array) }, o.squaredDistance = o.sqrDist, o.sqrLen = function(e) { return a.sqrLen(e._array) }, o.squaredLength = o.sqrLen, o.sub = function(e, t, r) { return a.subtract(e._array, t._array, r._array), e._dirty = !0, e }, o.subtract = o.sub, o.transformMat3 = function(e, t, r) { return a.transformMat3(e._array, t._array, r._array), e._dirty = !0, e }, o.transformMat4 = function(e, t, r) { return a.transformMat4(e._array, t._array, r._array), e._dirty = !0, e }, o.transformQuat = function(e, t, r) { return a.transformQuat(e._array, t._array, r._array), e._dirty = !0, e };
        var h = Math.atan2,
            l = Math.asin,
            c = Math.abs;
        o.eulerFromQuat = function(e, t, r) {
            e._dirty = !0, t = t._array;
            var i = e._array,
                a = t[0],
                o = t[1],
                s = t[2],
                u = t[3],
                c = a * a,
                d = o * o,
                f = s * s,
                p = u * u,
                r = (r || "XYZ").toUpperCase();
            switch (r) {
                case "XYZ":
                    i[0] = h(2 * (a * u - o * s), p - c - d + f), i[1] = l(n(2 * (a * s + o * u), -1, 1)), i[2] = h(2 * (s * u - a * o), p + c - d - f);
                    break;
                case "YXZ":
                    i[0] = l(n(2 * (a * u - o * s), -1, 1)), i[1] = h(2 * (a * s + o * u), p - c - d + f), i[2] = h(2 * (a * o + s * u), p - c + d - f);
                    break;
                case "ZXY":
                    i[0] = l(n(2 * (a * u + o * s), -1, 1)), i[1] = h(2 * (o * u - s * a), p - c - d + f), i[2] = h(2 * (s * u - a * o), p - c + d - f);
                    break;
                case "ZYX":
                    i[0] = h(2 * (a * u + s * o), p - c - d + f), i[1] = l(n(2 * (o * u - a * s), -1, 1)), i[2] = h(2 * (a * o + s * u), p + c - d - f);
                    break;
                case "YZX":
                    i[0] = h(2 * (a * u - s * o), p - c + d - f), i[1] = h(2 * (o * u - a * s), p + c - d - f), i[2] = l(n(2 * (a * o + s * u), -1, 1));
                    break;
                case "XZY":
                    i[0] = h(2 * (a * u + o * s), p - c + d - f), i[1] = h(2 * (a * s + o * u), p + c - d - f), i[2] = l(n(2 * (s * u - a * o), -1, 1));
                    break;
                default:
                    console.warn("Unkown order: " + r)
            }
            return e
        }, o.eulerFromMat3 = function(e, t, r) {
            var i = t._array,
                a = i[0],
                o = i[3],
                s = i[6],
                u = i[1],
                d = i[4],
                f = i[7],
                p = i[2],
                _ = i[5],
                m = i[8],
                g = e._array,
                r = (r || "XYZ").toUpperCase();
            switch (r) {
                case "XYZ":
                    g[1] = l(n(s, -1, 1)), c(s) < .99999 ? (g[0] = h(-f, m), g[2] = h(-o, a)) : (g[0] = h(_, d), g[2] = 0);
                    break;
                case "YXZ":
                    g[0] = l(-n(f, -1, 1)), c(f) < .99999 ? (g[1] = h(s, m), g[2] = h(u, d)) : (g[1] = h(-p, a), g[2] = 0);
                    break;
                case "ZXY":
                    g[0] = l(n(_, -1, 1)), c(_) < .99999 ? (g[1] = h(-p, m), g[2] = h(-o, d)) : (g[1] = 0, g[2] = h(u, a));
                    break;
                case "ZYX":
                    g[1] = l(-n(p, -1, 1)), c(p) < .99999 ? (g[0] = h(_, m), g[2] = h(u, a)) : (g[0] = 0, g[2] = h(-o, d));
                    break;
                case "YZX":
                    g[2] = l(n(u, -1, 1)), c(u) < .99999 ? (g[0] = h(-f, d), g[1] = h(-p, a)) : (g[0] = 0, g[1] = h(s, m));
                    break;
                case "XZY":
                    g[2] = l(-n(o, -1, 1)), c(o) < .99999 ? (g[0] = h(_, d), g[1] = h(s, a)) : (g[0] = h(-f, m), g[1] = 0);
                    break;
                default:
                    console.warn("Unkown order: " + r)
            }
            return e._dirty = !0, e
        }, o.POSITIVE_X = new o(1, 0, 0), o.NEGATIVE_X = new o(-1, 0, 0), o.POSITIVE_Y = new o(0, 1, 0), o.NEGATIVE_Y = new o(0, -1, 0), o.POSITIVE_Z = new o(0, 0, 1), o.NEGATIVE_Z = new o(0, 0, -1), o.UP = new o(0, 1, 0), o.ZERO = new o(0, 0, 0), e.exports = o
    }, function(e, t, r) {
        var n = r(0),
            i = {
                firstNotNull: function() {
                    for (var e = 0, t = arguments.length; e < t; e++)
                        if (null != arguments[e]) return arguments[e]
                },
                queryDataIndex: function(e, t) { return null != t.dataIndexInside ? t.dataIndexInside : null != t.dataIndex ? n.util.isArray(t.dataIndex) ? n.util.map(t.dataIndex, function(t) { return e.indexOfRawIndex(t) }) : e.indexOfRawIndex(t.dataIndex) : null != t.name ? n.util.isArray(t.name) ? n.util.map(t.name, function(t) { return e.indexOfName(t) }) : e.indexOfName(t.name) : void 0 }
            };
        e.exports = i
    }, function(e, t, r) {
        var n = r(6),
            i = r(17),
            a = r(11),
            o = r(80),
            s = o.isPowerOfTwo,
            u = n.extend(function() { return { image: null, pixels: null, mipmaps: [] } }, {
                update: function(e) {
                    e.bindTexture(e.TEXTURE_2D, this._cache.get("webgl_texture")), this.updateCommon(e);
                    var t = this.format,
                        r = this.type;
                    e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, this.wrapS), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, this.wrapT), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, this.magFilter), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, this.minFilter);
                    var n = i.getExtension(e, "EXT_texture_filter_anisotropic");
                    if (n && this.anisotropic > 1 && e.texParameterf(e.TEXTURE_2D, n.TEXTURE_MAX_ANISOTROPY_EXT, this.anisotropic), 36193 === r) { i.getExtension(e, "OES_texture_half_float") || (r = a.FLOAT) }
                    if (this.mipmaps.length)
                        for (var o = this.width, s = this.height, u = 0; u < this.mipmaps.length; u++) {
                            var h = this.mipmaps[u];
                            this._updateTextureData(e, h, u, o, s, t, r), o /= 2, s /= 2
                        } else this._updateTextureData(e, this, 0, this.width, this.height, t, r), this.useMipmap && !this.NPOT && e.generateMipmap(e.TEXTURE_2D);
                    e.bindTexture(e.TEXTURE_2D, null)
                },
                _updateTextureData: function(e, t, r, i, a, o, s) { t.image ? e.texImage2D(e.TEXTURE_2D, r, o, o, s, t.image) : o <= n.COMPRESSED_RGBA_S3TC_DXT5_EXT && o >= n.COMPRESSED_RGB_S3TC_DXT1_EXT ? e.compressedTexImage2D(e.TEXTURE_2D, r, o, i, a, 0, t.pixels) : e.texImage2D(e.TEXTURE_2D, r, o, i, a, 0, o, s, t.pixels) },
                generateMipmap: function(e) { this.useMipmap && !this.NPOT && (e.bindTexture(e.TEXTURE_2D, this._cache.get("webgl_texture")), e.generateMipmap(e.TEXTURE_2D)) },
                isPowerOfTwo: function() { var e, t; return this.image ? (e = this.image.width, t = this.image.height) : (e = this.width, t = this.height), s(e) && s(t) },
                isRenderable: function() { return this.image ? "CANVAS" === this.image.nodeName || "VIDEO" === this.image.nodeName || this.image.complete : !(!this.width || !this.height) },
                bind: function(e) { e.bindTexture(e.TEXTURE_2D, this.getWebGLTexture(e)) },
                unbind: function(e) { e.bindTexture(e.TEXTURE_2D, null) },
                load: function(e, t) {
                    var r = new Image;
                    t && (r.crossOrigin = t);
                    var n = this;
                    return r.onload = function() { n.dirty(), n.trigger("success", n), r.onload = null }, r.onerror = function() { n.trigger("error", n), r.onerror = null }, r.src = e, this.image = r, this
                }
            });
        Object.defineProperty(u.prototype, "width", { get: function() { return this.image ? this.image.width : this._width }, set: function(e) { this.image ? console.warn("Texture from image can't set width") : (this._width !== e && this.dirty(), this._width = e) } }), Object.defineProperty(u.prototype, "height", { get: function() { return this.image ? this.image.height : this._height }, set: function(e) { this.image ? console.warn("Texture from image can't set height") : (this._height !== e && this.dirty(), this._height = e) } }), e.exports = u
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(11),
            a = r(45),
            o = r(17),
            s = n.extend({ width: 512, height: 512, type: i.UNSIGNED_BYTE, format: i.RGBA, wrapS: i.CLAMP_TO_EDGE, wrapT: i.CLAMP_TO_EDGE, minFilter: i.LINEAR_MIPMAP_LINEAR, magFilter: i.LINEAR, useMipmap: !0, anisotropic: 1, flipY: !0, unpackAlignment: 4, premultiplyAlpha: !1, dynamic: !1, NPOT: !1 }, function() { this._cache = new a }, {
                getWebGLTexture: function(e) { var t = this._cache; return t.use(e.__GLID__), t.miss("webgl_texture") && t.put("webgl_texture", e.createTexture()), this.dynamic ? this.update(e) : t.isDirty() && (this.update(e), t.fresh()), t.get("webgl_texture") },
                bind: function() {},
                unbind: function() {},
                dirty: function() { this._cache && this._cache.dirtyAll() },
                update: function(e) {},
                updateCommon: function(e) { e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, this.flipY), e.pixelStorei(e.UNPACK_PREMULTIPLY_ALPHA_WEBGL, this.premultiplyAlpha), e.pixelStorei(e.UNPACK_ALIGNMENT, this.unpackAlignment), this._fallBack(e) },
                _fallBack: function(e) {
                    var t = this.isPowerOfTwo();
                    this.format === i.DEPTH_COMPONENT && (this.useMipmap = !1);
                    var r = o.getExtension(e, "EXT_sRGB");
                    this.format !== s.SRGB || r || (this.format = s.RGB), this.format !== s.SRGB_ALPHA || r || (this.format = s.RGBA), t && this.useMipmap ? (this.NPOT = !1, this._minFilterOriginal && (this.minFilter = this._minFilterOriginal), this._magFilterOriginal && (this.magFilter = this._magFilterOriginal), this._wrapSOriginal && (this.wrapS = this._wrapSOriginal), this._wrapTOriginal && (this.wrapT = this._wrapTOriginal)) : (this.NPOT = !0, this._minFilterOriginal = this.minFilter, this._magFilterOriginal = this.magFilter, this._wrapSOriginal = this.wrapS, this._wrapTOriginal = this.wrapT, this.minFilter == i.NEAREST_MIPMAP_NEAREST || this.minFilter == i.NEAREST_MIPMAP_LINEAR ? this.minFilter = i.NEAREST : this.minFilter != i.LINEAR_MIPMAP_LINEAR && this.minFilter != i.LINEAR_MIPMAP_NEAREST || (this.minFilter = i.LINEAR), this.wrapS = i.CLAMP_TO_EDGE, this.wrapT = i.CLAMP_TO_EDGE)
                },
                nextHighestPowerOfTwo: function(e) {--e; for (var t = 1; t < 32; t <<= 1) e |= e >> t; return e + 1 },
                dispose: function(e) {
                    var t = this._cache;
                    t.use(e.__GLID__);
                    var r = t.get("webgl_texture");
                    r && e.deleteTexture(r), t.deleteContext(e.__GLID__)
                },
                isRenderable: function() {},
                isPowerOfTwo: function() {}
            });
        Object.defineProperty(s.prototype, "width", { get: function() { return this._width }, set: function(e) { this._width = e } }), Object.defineProperty(s.prototype, "height", { get: function() { return this._height }, set: function(e) { this._height = e } }), s.BYTE = i.BYTE, s.UNSIGNED_BYTE = i.UNSIGNED_BYTE, s.SHORT = i.SHORT, s.UNSIGNED_SHORT = i.UNSIGNED_SHORT, s.INT = i.INT, s.UNSIGNED_INT = i.UNSIGNED_INT, s.FLOAT = i.FLOAT, s.HALF_FLOAT = 36193, s.UNSIGNED_INT_24_8_WEBGL = 34042, s.DEPTH_COMPONENT = i.DEPTH_COMPONENT, s.DEPTH_STENCIL = i.DEPTH_STENCIL, s.ALPHA = i.ALPHA, s.RGB = i.RGB, s.RGBA = i.RGBA, s.LUMINANCE = i.LUMINANCE, s.LUMINANCE_ALPHA = i.LUMINANCE_ALPHA, s.SRGB = 35904, s.SRGB_ALPHA = 35906, s.COMPRESSED_RGB_S3TC_DXT1_EXT = 33776, s.COMPRESSED_RGBA_S3TC_DXT1_EXT = 33777, s.COMPRESSED_RGBA_S3TC_DXT3_EXT = 33778, s.COMPRESSED_RGBA_S3TC_DXT5_EXT = 33779, s.NEAREST = i.NEAREST, s.LINEAR = i.LINEAR, s.NEAREST_MIPMAP_NEAREST = i.NEAREST_MIPMAP_NEAREST, s.LINEAR_MIPMAP_NEAREST = i.LINEAR_MIPMAP_NEAREST, s.NEAREST_MIPMAP_LINEAR = i.NEAREST_MIPMAP_LINEAR, s.LINEAR_MIPMAP_LINEAR = i.LINEAR_MIPMAP_LINEAR, s.REPEAT = i.REPEAT, s.CLAMP_TO_EDGE = i.CLAMP_TO_EDGE, s.MIRRORED_REPEAT = i.MIRRORED_REPEAT, e.exports = s
    }, function(e, t, r) {
        "use strict";

        function n() { return { locations: {}, attriblocations: {} } }

        function i(e, t, r) { if (!e.getShaderParameter(t, e.COMPILE_STATUS)) return [e.getShaderInfoLog(t), a(r)].join("\n") }

        function a(e) { for (var t = e.split("\n"), r = 0, n = t.length; r < n; r++) t[r] = r + 1 + ": " + t[r]; return t.join("\n") }
        var o = r(8),
            s = r(27),
            u = r(45),
            h = r(20),
            l = r(1),
            c = (r(17), l.mat2),
            d = l.mat3,
            f = l.mat4,
            p = /uniform\s+(bool|float|int|vec2|vec3|vec4|ivec2|ivec3|ivec4|mat2|mat3|mat4|sampler2D|samplerCube)\s+([\w\,]+)?(\[.*?\])?\s*(:\s*([\S\s]+?))?;/g,
            _ = /#define\s+(\w+)?(\s+[\w-.]+)?\s*;?\s*\n/g,
            m = { bool: "1i", int: "1i", sampler2D: "t", samplerCube: "t", float: "1f", vec2: "2f", vec3: "3f", vec4: "4f", ivec2: "2i", ivec3: "3i", ivec4: "4i", mat2: "m2", mat3: "m3", mat4: "m4" },
            g = { bool: function() { return !0 }, int: function() { return 0 }, float: function() { return 0 }, sampler2D: function() { return null }, samplerCube: function() { return null }, vec2: function() { return [0, 0] }, vec3: function() { return [0, 0, 0] }, vec4: function() { return [0, 0, 0, 0] }, ivec2: function() { return [0, 0] }, ivec3: function() { return [0, 0, 0] }, ivec4: function() { return [0, 0, 0, 0] }, mat2: function() { return c.create() }, mat3: function() { return d.create() }, mat4: function() { return f.create() }, array: function() { return [] } },
            v = ["POSITION", "NORMAL", "BINORMAL", "TANGENT", "TEXCOORD", "TEXCOORD_0", "TEXCOORD_1", "COLOR", "JOINT", "WEIGHT"],
            y = ["SKIN_MATRIX", "VIEWPORT_SIZE", "VIEWPORT", "DEVICEPIXELRATIO", "WINDOW_SIZE", "NEAR", "FAR", "TIME"],
            x = ["WORLD", "VIEW", "PROJECTION", "WORLDVIEW", "VIEWPROJECTION", "WORLDVIEWPROJECTION", "WORLDINVERSE", "VIEWINVERSE", "PROJECTIONINVERSE", "WORLDVIEWINVERSE", "VIEWPROJECTIONINVERSE", "WORLDVIEWPROJECTIONINVERSE", "WORLDTRANSPOSE", "VIEWTRANSPOSE", "PROJECTIONTRANSPOSE", "WORLDVIEWTRANSPOSE", "VIEWPROJECTIONTRANSPOSE", "WORLDVIEWPROJECTIONTRANSPOSE", "WORLDINVERSETRANSPOSE", "VIEWINVERSETRANSPOSE", "PROJECTIONINVERSETRANSPOSE", "WORLDVIEWINVERSETRANSPOSE", "VIEWPROJECTIONINVERSETRANSPOSE", "WORLDVIEWPROJECTIONINVERSETRANSPOSE"],
            T = {},
            b = o.extend(function() { return { vertex: "", fragment: "", precision: "highp", attribSemantics: {}, matrixSemantics: {}, uniformSemantics: {}, matrixSemanticKeys: [], uniformTemplates: {}, attributeTemplates: {}, vertexDefines: {}, fragmentDefines: {}, extensions: ["OES_standard_derivatives", "EXT_shader_texture_lod"], lightGroup: 0, lightNumber: {}, _textureSlot: 0, _attacheMaterialNumber: 0, _uniformList: [], _textureStatus: {}, _vertexProcessed: "", _fragmentProcessed: "", _currentLocationsMap: {} } }, function() { this._cache = new u, this._codeDirty = !0, this._updateShaderString() }, {
                isEqual: function(e) { return !!e && (this === e ? !this._codeDirty : (e._codeDirty && e._updateShaderString(), this._codeDirty && this._updateShaderString(), !(e._vertexProcessed !== this._vertexProcessed || e._fragmentProcessed !== this._fragmentProcessed))) },
                setVertex: function(e) { this.vertex = e, this._updateShaderString(), this.dirty() },
                setFragment: function(e) { this.fragment = e, this._updateShaderString(), this.dirty() },
                bind: function(e) {
                    var t = this._cache;
                    if (t.use(e.__GLID__, n), this._currentLocationsMap = t.get("locations"), this._textureSlot = 0, this._codeDirty && this._updateShaderString(), t.isDirty("program")) { var r = this._buildProgram(e, this._vertexProcessed, this._fragmentProcessed); if (t.fresh("program"), r) return r }
                    e.useProgram(t.get("program"))
                },
                dirty: function() {
                    var e = this._cache;
                    this._codeDirty = !0, e.dirtyAll("program");
                    for (var t = 0; t < e._caches.length; t++)
                        if (e._caches[t]) {
                            var r = e._caches[t];
                            r.locations = {}, r.attriblocations = {}
                        }
                },
                _updateShaderString: function(e) { this.vertex === this._vertexPrev && this.fragment === this._fragmentPrev || (this._parseImport(), this.attribSemantics = {}, this.matrixSemantics = {}, this._textureStatus = {}, this._parseUniforms(), this._parseAttributes(), this._parseDefines(), this._vertexPrev = this.vertex, this._fragmentPrev = this.fragment), this._addDefineExtensionAndPrecision(e), this._vertexProcessed = this._unrollLoop(this._vertexProcessed, this.vertexDefines), this._fragmentProcessed = this._unrollLoop(this._fragmentProcessed, this.fragmentDefines), this._codeDirty = !1 },
                define: function(e, t, r) {
                    var n = this.vertexDefines,
                        i = this.fragmentDefines;
                    "vertex" !== e && "fragment" !== e && "both" !== e && arguments.length < 3 && (r = t, t = e, e = "both"), r = null != r ? r : null, "vertex" !== e && "both" !== e || n[t] !== r && (n[t] = r, this.dirty()), "fragment" !== e && "both" !== e || i[t] !== r && (i[t] = r, "both" !== e && this.dirty())
                },
                undefine: function(e, t) { "vertex" !== e && "fragment" !== e && "both" !== e && arguments.length < 2 && (t = e, e = "both"), "vertex" !== e && "both" !== e || this.isDefined("vertex", t) && (delete this.vertexDefines[t], this.dirty()), "fragment" !== e && "both" !== e || this.isDefined("fragment", t) && (delete this.fragmentDefines[t], "both" !== e && this.dirty()) },
                isDefined: function(e, t) {
                    switch (e) {
                        case "vertex":
                            return void 0 !== this.vertexDefines[t];
                        case "fragment":
                            return void 0 !== this.fragmentDefines[t]
                    }
                },
                getDefine: function(e, t) {
                    switch (e) {
                        case "vertex":
                            return this.vertexDefines[t];
                        case "fragment":
                            return this.fragmentDefines[t]
                    }
                },
                enableTexture: function(e) {
                    if (e instanceof Array)
                        for (var t = 0; t < e.length; t++) this.enableTexture(e[t]);
                    else { var r = this._textureStatus[e]; if (r) { r.enabled || (r.enabled = !0, this.dirty()) } }
                },
                enableTexturesAll: function() {
                    var e = this._textureStatus;
                    for (var t in e) e[t].enabled = !0;
                    this.dirty()
                },
                disableTexture: function(e) {
                    if (e instanceof Array)
                        for (var t = 0; t < e.length; t++) this.disableTexture(e[t]);
                    else { var r = this._textureStatus[e]; if (r) {!r.enabled || (r.enabled = !1, this.dirty()) } }
                },
                disableTexturesAll: function() {
                    var e = this._textureStatus;
                    for (var t in e) e[t].enabled = !1;
                    this.dirty()
                },
                isTextureEnabled: function(e) { var t = this._textureStatus; return !!t[e] && t[e].enabled },
                getEnabledTextures: function() {
                    var e = [],
                        t = this._textureStatus;
                    for (var r in t) t[r].enabled && e.push(r);
                    return e
                },
                hasUniform: function(e) { var t = this._currentLocationsMap[e]; return null !== t && void 0 !== t },
                currentTextureSlot: function() { return this._textureSlot },
                resetTextureSlot: function(e) { this._textureSlot = e || 0 },
                takeCurrentTextureSlot: function(e, t) { var r = this._textureSlot; return this.useTextureSlot(e, t, r), this._textureSlot++, r },
                useTextureSlot: function(e, t, r) { t && (e.activeTexture(e.TEXTURE0 + r), t.isRenderable() ? t.bind(e) : t.unbind(e)) },
                setUniform: function(e, t, r, n) {
                    var i = this._currentLocationsMap,
                        a = i[r];
                    if (null === a || void 0 === a) return !1;
                    switch (t) {
                        case "m4":
                            e.uniformMatrix4fv(a, !1, n);
                            break;
                        case "2i":
                            e.uniform2i(a, n[0], n[1]);
                            break;
                        case "2f":
                            e.uniform2f(a, n[0], n[1]);
                            break;
                        case "3i":
                            e.uniform3i(a, n[0], n[1], n[2]);
                            break;
                        case "3f":
                            e.uniform3f(a, n[0], n[1], n[2]);
                            break;
                        case "4i":
                            e.uniform4i(a, n[0], n[1], n[2], n[3]);
                            break;
                        case "4f":
                            e.uniform4f(a, n[0], n[1], n[2], n[3]);
                            break;
                        case "1i":
                            e.uniform1i(a, n);
                            break;
                        case "1f":
                            e.uniform1f(a, n);
                            break;
                        case "1fv":
                            e.uniform1fv(a, n);
                            break;
                        case "1iv":
                            e.uniform1iv(a, n);
                            break;
                        case "2iv":
                            e.uniform2iv(a, n);
                            break;
                        case "2fv":
                            e.uniform2fv(a, n);
                            break;
                        case "3iv":
                            e.uniform3iv(a, n);
                            break;
                        case "3fv":
                            e.uniform3fv(a, n);
                            break;
                        case "4iv":
                            e.uniform4iv(a, n);
                            break;
                        case "4fv":
                            e.uniform4fv(a, n);
                            break;
                        case "m2":
                        case "m2v":
                            e.uniformMatrix2fv(a, !1, n);
                            break;
                        case "m3":
                        case "m3v":
                            e.uniformMatrix3fv(a, !1, n);
                            break;
                        case "m4v":
                            if (n instanceof Array) {
                                for (var o = new h.Float32Array(16 * n.length), s = 0, u = 0; u < n.length; u++)
                                    for (var l = n[u], c = 0; c < 16; c++) o[s++] = l[c];
                                e.uniformMatrix4fv(a, !1, o)
                            } else n instanceof h.Float32Array && e.uniformMatrix4fv(a, !1, n)
                    }
                    return !0
                },
                setUniformOfSemantic: function(e, t, r) { var n = this.uniformSemantics[t]; return !!n && this.setUniform(e, n.type, n.symbol, r) },
                enableAttributes: function(e, t, r) {
                    var n, i = this._cache.get("program"),
                        a = this._cache.get("attriblocations");
                    (n = r ? r.__enabledAttributeList : T[e.__GLID__]) || (n = r ? r.__enabledAttributeList = [] : T[e.__GLID__] = []);
                    for (var o = [], s = 0; s < t.length; s++) {
                        var u = t[s];
                        if (this.attributeTemplates[u]) {
                            var h = a[u];
                            if (void 0 === h) {
                                if (-1 === (h = e.getAttribLocation(i, u))) { o[s] = -1; continue }
                                a[u] = h
                            }
                            o[s] = h, n[h] ? n[h] = 2 : n[h] = 1
                        } else o[s] = -1
                    }
                    for (var s = 0; s < n.length; s++) switch (n[s]) {
                        case 1:
                            e.enableVertexAttribArray(s), n[s] = 3;
                            break;
                        case 2:
                            n[s] = 3;
                            break;
                        case 3:
                            e.disableVertexAttribArray(s), n[s] = 0
                    }
                    return o
                },
                _parseImport: function() { this._vertexProcessedWithoutDefine = b.parseImport(this.vertex), this._fragmentProcessedWithoutDefine = b.parseImport(this.fragment) },
                _addDefineExtensionAndPrecision: function(e) {
                    e = e || this.extensions;
                    for (var t = [], r = 0; r < e.length; r++) t.push("#extension GL_" + e[r] + " : enable");
                    var n = this._getDefineStr(this.vertexDefines);
                    this._vertexProcessed = n + "\n" + this._vertexProcessedWithoutDefine;
                    var n = this._getDefineStr(this.fragmentDefines),
                        i = n + "\n" + this._fragmentProcessedWithoutDefine;
                    this._fragmentProcessed = t.join("\n") + "\n" + ["precision", this.precision, "float"].join(" ") + ";\n" + ["precision", this.precision, "int"].join(" ") + ";\n" + ["precision", this.precision, "sampler2D"].join(" ") + ";\n" + i
                },
                _getDefineStr: function(e) {
                    var t = this.lightNumber,
                        r = this._textureStatus,
                        n = [];
                    for (var i in t) {
                        var a = t[i];
                        a > 0 && n.push("#define " + i.toUpperCase() + "_COUNT " + a)
                    }
                    for (var o in r) { r[o].enabled && n.push("#define " + o.toUpperCase() + "_ENABLED") }
                    for (var o in e) {
                        var s = e[o];
                        null === s ? n.push("#define " + o) : n.push("#define " + o + " " + s.toString())
                    }
                    return n.join("\n")
                },
                _unrollLoop: function(e, t) {
                    function r(e, r, i, a) {
                        var o = "";
                        isNaN(r) && (r = r in t ? t[r] : n[r]), isNaN(i) && (i = i in t ? t[i] : n[i]);
                        for (var s = parseInt(r); s < parseInt(i); s++) o += "{" + a.replace(/float\s*\(\s*_idx_\s*\)/g, s.toFixed(1)).replace(/_idx_/g, s) + "}";
                        return o
                    }
                    var n = {};
                    for (var i in this.lightNumber) n[i + "_COUNT"] = this.lightNumber[i];
                    return e.replace(/for\s*?\(int\s*?_idx_\s*\=\s*([\w-]+)\;\s*_idx_\s*<\s*([\w-]+);\s*_idx_\s*\+\+\s*\)\s*\{\{([\s\S]+?)(?=\}\})\}\}/g, r)
                },
                _parseUniforms: function() {
                    function e(e, i, a, o, s, u) {
                        if (i && a) {
                            var h, l = m[i],
                                c = !0;
                            if (l) {
                                if (r._uniformList.push(a), "sampler2D" !== i && "samplerCube" !== i || (r._textureStatus[a] = { enabled: !1, shaderType: n }), o && (l += "v"), u)
                                    if (v.indexOf(u) >= 0) r.attribSemantics[u] = { symbol: a, type: l }, c = !1;
                                    else if (x.indexOf(u) >= 0) {
                                    var d = !1,
                                        f = u;
                                    u.match(/TRANSPOSE$/) && (d = !0, f = u.slice(0, -9)), r.matrixSemantics[u] = { symbol: a, type: l, isTranspose: d, semanticNoTranspose: f }, c = !1
                                } else if (y.indexOf(u) >= 0) r.uniformSemantics[u] = { symbol: a, type: l }, c = !1;
                                else if ("unconfigurable" === u) c = !1;
                                else {
                                    if (!(h = r._parseDefaultValue(i, u))) throw new Error('Unkown semantic "' + u + '"');
                                    u = ""
                                }
                                c && (t[a] = { type: l, value: o ? g.array : h || g[i], semantic: u || null })
                            }
                            return ["uniform", i, a, o].join(" ") + ";\n"
                        }
                    }
                    var t = {},
                        r = this,
                        n = "vertex";
                    this._uniformList = [], this._vertexProcessedWithoutDefine = this._vertexProcessedWithoutDefine.replace(p, e), n = "fragment", this._fragmentProcessedWithoutDefine = this._fragmentProcessedWithoutDefine.replace(p, e), r.matrixSemanticKeys = Object.keys(this.matrixSemantics), this.uniformTemplates = t
                },
                _parseDefaultValue: function(e, t) { var r = /\[\s*(.*)\s*\]/; { if ("vec2" !== e && "vec3" !== e && "vec4" !== e) return "bool" === e ? function() { return "true" === t.toLowerCase() } : "float" === e ? function() { return parseFloat(t) } : "int" === e ? function() { return parseInt(t) } : void 0; var n = r.exec(t)[1]; if (n) { var i = n.split(/\s*,\s*/); return function() { return new h.Float32Array(i) } } } },
                createUniforms: function() {
                    var e = {};
                    for (var t in this.uniformTemplates) {
                        var r = this.uniformTemplates[t];
                        e[t] = { type: r.type, value: r.value() }
                    }
                    return e
                },
                attached: function() { this._attacheMaterialNumber++ },
                detached: function() { this._attacheMaterialNumber-- },
                isAttachedToAny: function() { return 0 !== this._attacheMaterialNumber },
                _parseAttributes: function() {
                    function e(e, n, i, a, o) {
                        if (n && i) {
                            var s = 1;
                            switch (n) {
                                case "vec4":
                                    s = 4;
                                    break;
                                case "vec3":
                                    s = 3;
                                    break;
                                case "vec2":
                                    s = 2;
                                    break;
                                case "float":
                                    s = 1
                            }
                            if (t[i] = { type: "float", size: s, semantic: o || null }, o) {
                                if (v.indexOf(o) < 0) throw new Error('Unkown semantic "' + o + '"');
                                r.attribSemantics[o] = { symbol: i, type: n }
                            }
                        }
                        return ["attribute", n, i].join(" ") + ";\n"
                    }
                    var t = {},
                        r = this;
                    this._vertexProcessedWithoutDefine = this._vertexProcessedWithoutDefine.replace(/attribute\s+(float|int|vec2|vec3|vec4)\s+(\w*)\s*(:\s*(\w+))?;/g, e), this.attributeTemplates = t
                },
                _parseDefines: function() {
                    function e(e, n, i) { var a = "vertex" === r ? t.vertexDefines : t.fragmentDefines; return a[n] || (a[n] = "false" != i && ("true" == i || (i ? isNaN(parseFloat(i)) ? i : parseFloat(i) : null))), "" }
                    var t = this,
                        r = "vertex";
                    this._vertexProcessedWithoutDefine = this._vertexProcessedWithoutDefine.replace(_, e), r = "fragment", this._fragmentProcessedWithoutDefine = this._fragmentProcessedWithoutDefine.replace(_, e)
                },
                _buildProgram: function(e, t, r) {
                    var n = this._cache;
                    n.get("program") && e.deleteProgram(n.get("program"));
                    var a = e.createProgram(),
                        o = e.createShader(e.VERTEX_SHADER);
                    e.shaderSource(o, t), e.compileShader(o);
                    var s = e.createShader(e.FRAGMENT_SHADER);
                    e.shaderSource(s, r), e.compileShader(s);
                    var u = i(e, o, t);
                    if (u) return u;
                    if (u = i(e, s, r)) return u;
                    if (e.attachShader(a, o), e.attachShader(a, s), this.attribSemantics.POSITION) e.bindAttribLocation(a, 0, this.attribSemantics.POSITION.symbol);
                    else {
                        var h = Object.keys(this.attributeTemplates);
                        e.bindAttribLocation(a, 0, h[0])
                    }
                    if (e.linkProgram(a), !e.getProgramParameter(a, e.LINK_STATUS)) return "Could not link program\nVALIDATE_STATUS: " + e.getProgramParameter(a, e.VALIDATE_STATUS) + ", gl error [" + e.getError() + "]";
                    for (var l = 0; l < this._uniformList.length; l++) {
                        var c = this._uniformList[l];
                        n.get("locations")[c] = e.getUniformLocation(a, c)
                    }
                    e.deleteShader(o), e.deleteShader(s), n.put("program", a)
                },
                clone: function() { var e = new b({ vertex: this.vertex, fragment: this.fragment, vertexDefines: s.clone(this.vertexDefines), fragmentDefines: s.clone(this.fragmentDefines) }); for (var t in this._textureStatus) e._textureStatus[t] = s.clone(this._textureStatus[t]); return e },
                dispose: function(e) {
                    var t = this._cache;
                    t.use(e.__GLID__);
                    var r = t.get("program");
                    r && e.deleteProgram(r), t.deleteContext(e.__GLID__), this._locations = {}
                }
            });
        b.parseImport = function(e) { return e = e.replace(/(@import)\s*([0-9a-zA-Z_\-\.]*)/g, function(e, t, r) { var e = b.source(r); return e ? b.parseImport(e) : (console.error('Shader chunk "' + r + '" not existed in library'), "") }) };
        b.import = function(e) {
            e.replace(/(@export)\s*([0-9a-zA-Z_\-\.]*)\s*\n([\s\S]*?)@end/g, function(e, t, r, n) {
                var n = n.replace(/(^[\s\t\xa0\u3000]+)|([\u3000\xa0\s\t]+\x24)/g, "");
                if (n) {
                    for (var i, a = r.split("."), o = b.codes, s = 0; s < a.length - 1;) i = a[s++], o[i] || (o[i] = {}), o = o[i];
                    i = a[s], o[i] = n
                }
                return n
            })
        }, b.codes = {}, b.source = function(e) { for (var t = e.split("."), r = b.codes, n = 0; r && n < t.length;) { r = r[t[n++]] } return "string" != typeof r ? (console.error('Shader "' + e + '" not existed in library'), "") : r }, e.exports = b
    }, function(e, t, r) {
        "use strict";
        var n = r(201),
            i = r(53),
            a = r(27),
            o = function() { this.__GUID__ = a.genGUID() };
        o.__initializers__ = [function(e) { a.extend(this, e) }], a.extend(o, n), a.extend(o.prototype, i), e.exports = o
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = r(3),
            a = n.mat4,
            o = n.vec3,
            s = n.mat3,
            u = n.quat,
            h = function() { this._axisX = new i, this._axisY = new i, this._axisZ = new i, this._array = a.create(), this._dirty = !0 };
        h.prototype = {
            constructor: h,
            setArray: function(e) { for (var t = 0; t < this._array.length; t++) this._array[t] = e[t]; return this._dirty = !0, this },
            adjoint: function() { return a.adjoint(this._array, this._array), this._dirty = !0, this },
            clone: function() { return (new h).copy(this) },
            copy: function(e) { return a.copy(this._array, e._array), this._dirty = !0, this },
            determinant: function() { return a.determinant(this._array) },
            fromQuat: function(e) { return a.fromQuat(this._array, e._array), this._dirty = !0, this },
            fromRotationTranslation: function(e, t) { return a.fromRotationTranslation(this._array, e._array, t._array), this._dirty = !0, this },
            fromMat2d: function(e) { return h.fromMat2d(this, e), this },
            frustum: function(e, t, r, n, i, o) { return a.frustum(this._array, e, t, r, n, i, o), this._dirty = !0, this },
            identity: function() { return a.identity(this._array), this._dirty = !0, this },
            invert: function() { return a.invert(this._array, this._array), this._dirty = !0, this },
            lookAt: function(e, t, r) { return a.lookAt(this._array, e._array, t._array, r._array), this._dirty = !0, this },
            mul: function(e) { return a.mul(this._array, this._array, e._array), this._dirty = !0, this },
            mulLeft: function(e) { return a.mul(this._array, e._array, this._array), this._dirty = !0, this },
            multiply: function(e) { return a.multiply(this._array, this._array, e._array), this._dirty = !0, this },
            multiplyLeft: function(e) { return a.multiply(this._array, e._array, this._array), this._dirty = !0, this },
            ortho: function(e, t, r, n, i, o) { return a.ortho(this._array, e, t, r, n, i, o), this._dirty = !0, this },
            perspective: function(e, t, r, n) { return a.perspective(this._array, e, t, r, n), this._dirty = !0, this },
            rotate: function(e, t) { return a.rotate(this._array, this._array, e, t._array), this._dirty = !0, this },
            rotateX: function(e) { return a.rotateX(this._array, this._array, e), this._dirty = !0, this },
            rotateY: function(e) { return a.rotateY(this._array, this._array, e), this._dirty = !0, this },
            rotateZ: function(e) { return a.rotateZ(this._array, this._array, e), this._dirty = !0, this },
            scale: function(e) { return a.scale(this._array, this._array, e._array), this._dirty = !0, this },
            translate: function(e) { return a.translate(this._array, this._array, e._array), this._dirty = !0, this },
            transpose: function() { return a.transpose(this._array, this._array), this._dirty = !0, this },
            decomposeMatrix: function() {
                var e = o.create(),
                    t = o.create(),
                    r = o.create(),
                    n = s.create();
                return function(i, a, h) {
                    var l = this._array;
                    o.set(e, l[0], l[1], l[2]), o.set(t, l[4], l[5], l[6]), o.set(r, l[8], l[9], l[10]);
                    var c = o.length(e),
                        d = o.length(t),
                        f = o.length(r);
                    this.determinant() < 0 && (c = -c), i && i.set(c, d, f), h.set(l[12], l[13], l[14]), s.fromMat4(n, l), n[0] /= c, n[1] /= c, n[2] /= c, n[3] /= d, n[4] /= d, n[5] /= d, n[6] /= f, n[7] /= f, n[8] /= f, u.fromMat3(a._array, n), u.normalize(a._array, a._array), a._dirty = !0, h._dirty = !0
                }
            }(),
            toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" },
            toArray: function() { return Array.prototype.slice.call(this._array) }
        };
        var l = Object.defineProperty;
        if (l) {
            var c = h.prototype;
            l(c, "z", {
                get: function() { var e = this._array; return this._axisZ.set(e[8], e[9], e[10]), this._axisZ },
                set: function(e) {
                    var t = this._array;
                    e = e._array, t[8] = e[0], t[9] = e[1], t[10] = e[2], this._dirty = !0
                }
            }), l(c, "y", {
                get: function() { var e = this._array; return this._axisY.set(e[4], e[5], e[6]), this._axisY },
                set: function(e) {
                    var t = this._array;
                    e = e._array, t[4] = e[0], t[5] = e[1], t[6] = e[2], this._dirty = !0
                }
            }), l(c, "x", {
                get: function() { var e = this._array; return this._axisX.set(e[0], e[1], e[2]), this._axisX },
                set: function(e) {
                    var t = this._array;
                    e = e._array, t[0] = e[0], t[1] = e[1], t[2] = e[2], this._dirty = !0
                }
            })
        }
        h.adjoint = function(e, t) { return a.adjoint(e._array, t._array), e._dirty = !0, e }, h.copy = function(e, t) { return a.copy(e._array, t._array), e._dirty = !0, e }, h.determinant = function(e) { return a.determinant(e._array) }, h.identity = function(e) { return a.identity(e._array), e._dirty = !0, e }, h.ortho = function(e, t, r, n, i, o, s) { return a.ortho(e._array, t, r, n, i, o, s), e._dirty = !0, e }, h.perspective = function(e, t, r, n, i) { return a.perspective(e._array, t, r, n, i), e._dirty = !0, e }, h.lookAt = function(e, t, r, n) { return a.lookAt(e._array, t._array, r._array, n._array), e._dirty = !0, e }, h.invert = function(e, t) { return a.invert(e._array, t._array), e._dirty = !0, e }, h.mul = function(e, t, r) { return a.mul(e._array, t._array, r._array), e._dirty = !0, e }, h.multiply = h.mul, h.fromQuat = function(e, t) { return a.fromQuat(e._array, t._array), e._dirty = !0, e }, h.fromRotationTranslation = function(e, t, r) { return a.fromRotationTranslation(e._array, t._array, r._array), e._dirty = !0, e }, h.fromMat2d = function(e, t) {
            e._dirty = !0;
            var t = t._array,
                e = e._array;
            return e[0] = t[0], e[4] = t[2], e[12] = t[4], e[1] = t[1], e[5] = t[3], e[13] = t[5], e
        }, h.rotate = function(e, t, r, n) { return a.rotate(e._array, t._array, r, n._array), e._dirty = !0, e }, h.rotateX = function(e, t, r) { return a.rotateX(e._array, t._array, r), e._dirty = !0, e }, h.rotateY = function(e, t, r) { return a.rotateY(e._array, t._array, r), e._dirty = !0, e }, h.rotateZ = function(e, t, r) { return a.rotateZ(e._array, t._array, r), e._dirty = !0, e }, h.scale = function(e, t, r) { return a.scale(e._array, t._array, r._array), e._dirty = !0, e }, h.transpose = function(e, t) { return a.transpose(e._array, t._array), e._dirty = !0, e }, h.translate = function(e, t, r) { return a.translate(e._array, t._array, r._array), e._dirty = !0, e }, e.exports = h
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(6),
            a = r(23),
            o = r(17),
            s = r(11),
            u = r(45),
            h = s.FRAMEBUFFER,
            l = s.RENDERBUFFER,
            c = s.DEPTH_ATTACHMENT,
            d = s.COLOR_ATTACHMENT0,
            f = n.extend({ depthBuffer: !0, viewport: null, _width: 0, _height: 0, _textures: null, _boundRenderer: null }, function() { this._cache = new u, this._textures = {} }, {
                getTextureWidth: function() { return this._width },
                getTextureHeight: function() { return this._height },
                bind: function(e) {
                    if (e.__currentFrameBuffer) {
                        if (e.__currentFrameBuffer === this) return;
                        console.warn("Renderer already bound with another framebuffer. Unbind it first")
                    }
                    e.__currentFrameBuffer = this;
                    var t = e.gl;
                    t.bindFramebuffer(h, this._getFrameBufferGL(t)), this._boundRenderer = e;
                    var r = this._cache;
                    r.put("viewport", e.viewport);
                    var n, i, a = !1;
                    for (var o in this._textures) {
                        a = !0;
                        var s = this._textures[o];
                        s && (n = s.texture.width, i = s.texture.height, this._doAttach(t, s.texture, o, s.target))
                    }
                    this._width = n, this._height = i, !a && this.depthBuffer && console.error("Must attach texture before bind, or renderbuffer may have incorrect width and height."), this.viewport ? e.setViewport(this.viewport) : e.setViewport(0, 0, n, i, 1);
                    var u = r.get("attached_textures");
                    if (u)
                        for (var o in u)
                            if (!this._textures[o]) {
                                var d = u[o];
                                this._doDetach(t, o, d)
                            }
                    if (!r.get("depthtexture_attached") && this.depthBuffer) {
                        r.miss("renderbuffer") && r.put("renderbuffer", t.createRenderbuffer());
                        var f = r.get("renderbuffer");
                        n === r.get("renderbuffer_width") && i === r.get("renderbuffer_height") || (t.bindRenderbuffer(l, f), t.renderbufferStorage(l, t.DEPTH_COMPONENT16, n, i), r.put("renderbuffer_width", n), r.put("renderbuffer_height", i), t.bindRenderbuffer(l, null)), r.get("renderbuffer_attached") || (t.framebufferRenderbuffer(h, c, l, f), r.put("renderbuffer_attached", !0))
                    }
                },
                unbind: function(e) {
                    e.__currentFrameBuffer = null;
                    var t = e.gl;
                    t.bindFramebuffer(h, null), this._boundRenderer = null, this._cache.use(t.__GLID__);
                    var r = this._cache.get("viewport");
                    r && e.setViewport(r), this.updateMipmap(t)
                },
                updateMipmap: function(e) {
                    for (var t in this._textures) {
                        var r = this._textures[t];
                        if (r) {
                            var n = r.texture;
                            if (!n.NPOT && n.useMipmap && n.minFilter === i.LINEAR_MIPMAP_LINEAR) {
                                var o = n instanceof a ? s.TEXTURE_CUBE_MAP : s.TEXTURE_2D;
                                e.bindTexture(o, n.getWebGLTexture(e)), e.generateMipmap(o), e.bindTexture(o, null)
                            }
                        }
                    }
                },
                checkStatus: function(e) { return e.checkFramebufferStatus(h) },
                _getFrameBufferGL: function(e) { var t = this._cache; return t.use(e.__GLID__), t.miss("framebuffer") && t.put("framebuffer", e.createFramebuffer()), t.get("framebuffer") },
                attach: function(e, t, r) {
                    if (!e.width) throw new Error("The texture attached to color buffer is not a valid.");
                    t = t || d, r = r || s.TEXTURE_2D;
                    var n, i = this._boundRenderer,
                        a = i && i.gl;
                    if (a) {
                        var o = this._cache;
                        o.use(a.__GLID__), n = o.get("attached_textures")
                    }
                    var u = this._textures[t];
                    if (!u || u.target !== r || u.texture !== e || !n || null == n[t]) {
                        var h = !0;
                        a && (h = this._doAttach(a, e, t, r), this.viewport || i.setViewport(0, 0, e.width, e.height, 1)), h && (this._textures[t] = this._textures[t] || {}, this._textures[t].texture = e, this._textures[t].target = r)
                    }
                },
                _doAttach: function(e, t, r, n) {
                    var i = t.getWebGLTexture(e),
                        a = this._cache.get("attached_textures");
                    if (a && a[r]) { var u = a[r]; if (u.texture === t && u.target === n) return }
                    r = +r;
                    var d = !0;
                    if (r === c || r === s.DEPTH_STENCIL_ATTACHMENT) {
                        if (o.getExtension(e, "WEBGL_depth_texture") || (console.error("Depth texture is not supported by the browser"), d = !1), t.format !== s.DEPTH_COMPONENT && t.format !== s.DEPTH_STENCIL && (console.error("The texture attached to depth buffer is not a valid."), d = !1), d) {
                            var f = this._cache.get("renderbuffer");
                            f && (e.framebufferRenderbuffer(h, c, l, null), e.deleteRenderbuffer(f), this._cache.put("renderbuffer", !1)), this._cache.put("renderbuffer_attached", !1), this._cache.put("depthtexture_attached", !0)
                        }
                    }
                    return e.framebufferTexture2D(h, r, n, i, 0), a || (a = {}, this._cache.put("attached_textures", a)), a[r] = a[r] || {}, a[r].texture = t, a[r].target = n, d
                },
                _doDetach: function(e, t, r) {
                    e.framebufferTexture2D(h, t, r, null, 0);
                    var n = this._cache.get("attached_textures");
                    n && n[t] && (n[t] = null), t !== c && t !== s.DEPTH_STENCIL_ATTACHMENT || this._cache.put("depthtexture_attached", !1)
                },
                detach: function(e, t) {
                    if (this._textures[e] = null, this._boundRenderer) {
                        var r = this._boundRenderer.gl;
                        this._cache.use(r.__GLID__), this._doDetach(r, e, t)
                    }
                },
                dispose: function(e) {
                    var t = this._cache;
                    t.use(e.__GLID__);
                    var r = t.get("renderbuffer");
                    r && e.deleteRenderbuffer(r);
                    var n = t.get("framebuffer");
                    n && e.deleteFramebuffer(n), t.deleteContext(e.__GLID__), this._textures = {}
                }
            });
        f.DEPTH_ATTACHMENT = c, f.COLOR_ATTACHMENT0 = d, f.STENCIL_ATTACHMENT = s.STENCIL_ATTACHMENT, f.DEPTH_STENCIL_ATTACHMENT = s.DEPTH_STENCIL_ATTACHMENT, e.exports = f
    }, function(e, t) { e.exports = { DEPTH_BUFFER_BIT: 256, STENCIL_BUFFER_BIT: 1024, COLOR_BUFFER_BIT: 16384, POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4, TRIANGLE_STRIP: 5, TRIANGLE_FAN: 6, ZERO: 0, ONE: 1, SRC_COLOR: 768, ONE_MINUS_SRC_COLOR: 769, SRC_ALPHA: 770, ONE_MINUS_SRC_ALPHA: 771, DST_ALPHA: 772, ONE_MINUS_DST_ALPHA: 773, DST_COLOR: 774, ONE_MINUS_DST_COLOR: 775, SRC_ALPHA_SATURATE: 776, FUNC_ADD: 32774, BLEND_EQUATION: 32777, BLEND_EQUATION_RGB: 32777, BLEND_EQUATION_ALPHA: 34877, FUNC_SUBTRACT: 32778, FUNC_REVERSE_SUBTRACT: 32779, BLEND_DST_RGB: 32968, BLEND_SRC_RGB: 32969, BLEND_DST_ALPHA: 32970, BLEND_SRC_ALPHA: 32971, CONSTANT_COLOR: 32769, ONE_MINUS_CONSTANT_COLOR: 32770, CONSTANT_ALPHA: 32771, ONE_MINUS_CONSTANT_ALPHA: 32772, BLEND_COLOR: 32773, ARRAY_BUFFER: 34962, ELEMENT_ARRAY_BUFFER: 34963, ARRAY_BUFFER_BINDING: 34964, ELEMENT_ARRAY_BUFFER_BINDING: 34965, STREAM_DRAW: 35040, STATIC_DRAW: 35044, DYNAMIC_DRAW: 35048, BUFFER_SIZE: 34660, BUFFER_USAGE: 34661, CURRENT_VERTEX_ATTRIB: 34342, FRONT: 1028, BACK: 1029, FRONT_AND_BACK: 1032, CULL_FACE: 2884, BLEND: 3042, DITHER: 3024, STENCIL_TEST: 2960, DEPTH_TEST: 2929, SCISSOR_TEST: 3089, POLYGON_OFFSET_FILL: 32823, SAMPLE_ALPHA_TO_COVERAGE: 32926, SAMPLE_COVERAGE: 32928, NO_ERROR: 0, INVALID_ENUM: 1280, INVALID_VALUE: 1281, INVALID_OPERATION: 1282, OUT_OF_MEMORY: 1285, CW: 2304, CCW: 2305, LINE_WIDTH: 2849, ALIASED_POINT_SIZE_RANGE: 33901, ALIASED_LINE_WIDTH_RANGE: 33902, CULL_FACE_MODE: 2885, FRONT_FACE: 2886, DEPTH_RANGE: 2928, DEPTH_WRITEMASK: 2930, DEPTH_CLEAR_VALUE: 2931, DEPTH_FUNC: 2932, STENCIL_CLEAR_VALUE: 2961, STENCIL_FUNC: 2962, STENCIL_FAIL: 2964, STENCIL_PASS_DEPTH_FAIL: 2965, STENCIL_PASS_DEPTH_PASS: 2966, STENCIL_REF: 2967, STENCIL_VALUE_MASK: 2963, STENCIL_WRITEMASK: 2968, STENCIL_BACK_FUNC: 34816, STENCIL_BACK_FAIL: 34817, STENCIL_BACK_PASS_DEPTH_FAIL: 34818, STENCIL_BACK_PASS_DEPTH_PASS: 34819, STENCIL_BACK_REF: 36003, STENCIL_BACK_VALUE_MASK: 36004, STENCIL_BACK_WRITEMASK: 36005, VIEWPORT: 2978, SCISSOR_BOX: 3088, COLOR_CLEAR_VALUE: 3106, COLOR_WRITEMASK: 3107, UNPACK_ALIGNMENT: 3317, PACK_ALIGNMENT: 3333, MAX_TEXTURE_SIZE: 3379, MAX_VIEWPORT_DIMS: 3386, SUBPIXEL_BITS: 3408, RED_BITS: 3410, GREEN_BITS: 3411, BLUE_BITS: 3412, ALPHA_BITS: 3413, DEPTH_BITS: 3414, STENCIL_BITS: 3415, POLYGON_OFFSET_UNITS: 10752, POLYGON_OFFSET_FACTOR: 32824, TEXTURE_BINDING_2D: 32873, SAMPLE_BUFFERS: 32936, SAMPLES: 32937, SAMPLE_COVERAGE_VALUE: 32938, SAMPLE_COVERAGE_INVERT: 32939, COMPRESSED_TEXTURE_FORMATS: 34467, DONT_CARE: 4352, FASTEST: 4353, NICEST: 4354, GENERATE_MIPMAP_HINT: 33170, BYTE: 5120, UNSIGNED_BYTE: 5121, SHORT: 5122, UNSIGNED_SHORT: 5123, INT: 5124, UNSIGNED_INT: 5125, FLOAT: 5126, DEPTH_COMPONENT: 6402, ALPHA: 6406, RGB: 6407, RGBA: 6408, LUMINANCE: 6409, LUMINANCE_ALPHA: 6410, UNSIGNED_SHORT_4_4_4_4: 32819, UNSIGNED_SHORT_5_5_5_1: 32820, UNSIGNED_SHORT_5_6_5: 33635, FRAGMENT_SHADER: 35632, VERTEX_SHADER: 35633, MAX_VERTEX_ATTRIBS: 34921, MAX_VERTEX_UNIFORM_VECTORS: 36347, MAX_VARYING_VECTORS: 36348, MAX_COMBINED_TEXTURE_IMAGE_UNITS: 35661, MAX_VERTEX_TEXTURE_IMAGE_UNITS: 35660, MAX_TEXTURE_IMAGE_UNITS: 34930, MAX_FRAGMENT_UNIFORM_VECTORS: 36349, SHADER_TYPE: 35663, DELETE_STATUS: 35712, LINK_STATUS: 35714, VALIDATE_STATUS: 35715, ATTACHED_SHADERS: 35717, ACTIVE_UNIFORMS: 35718, ACTIVE_ATTRIBUTES: 35721, SHADING_LANGUAGE_VERSION: 35724, CURRENT_PROGRAM: 35725, NEVER: 512, LESS: 513, EQUAL: 514, LEQUAL: 515, GREATER: 516, NOTEQUAL: 517, GEQUAL: 518, ALWAYS: 519, KEEP: 7680, REPLACE: 7681, INCR: 7682, DECR: 7683, INVERT: 5386, INCR_WRAP: 34055, DECR_WRAP: 34056, VENDOR: 7936, RENDERER: 7937, VERSION: 7938, NEAREST: 9728, LINEAR: 9729, NEAREST_MIPMAP_NEAREST: 9984, LINEAR_MIPMAP_NEAREST: 9985, NEAREST_MIPMAP_LINEAR: 9986, LINEAR_MIPMAP_LINEAR: 9987, TEXTURE_MAG_FILTER: 10240, TEXTURE_MIN_FILTER: 10241, TEXTURE_WRAP_S: 10242, TEXTURE_WRAP_T: 10243, TEXTURE_2D: 3553, TEXTURE: 5890, TEXTURE_CUBE_MAP: 34067, TEXTURE_BINDING_CUBE_MAP: 34068, TEXTURE_CUBE_MAP_POSITIVE_X: 34069, TEXTURE_CUBE_MAP_NEGATIVE_X: 34070, TEXTURE_CUBE_MAP_POSITIVE_Y: 34071, TEXTURE_CUBE_MAP_NEGATIVE_Y: 34072, TEXTURE_CUBE_MAP_POSITIVE_Z: 34073, TEXTURE_CUBE_MAP_NEGATIVE_Z: 34074, MAX_CUBE_MAP_TEXTURE_SIZE: 34076, TEXTURE0: 33984, TEXTURE1: 33985, TEXTURE2: 33986, TEXTURE3: 33987, TEXTURE4: 33988, TEXTURE5: 33989, TEXTURE6: 33990, TEXTURE7: 33991, TEXTURE8: 33992, TEXTURE9: 33993, TEXTURE10: 33994, TEXTURE11: 33995, TEXTURE12: 33996, TEXTURE13: 33997, TEXTURE14: 33998, TEXTURE15: 33999, TEXTURE16: 34e3, TEXTURE17: 34001, TEXTURE18: 34002, TEXTURE19: 34003, TEXTURE20: 34004, TEXTURE21: 34005, TEXTURE22: 34006, TEXTURE23: 34007, TEXTURE24: 34008, TEXTURE25: 34009, TEXTURE26: 34010, TEXTURE27: 34011, TEXTURE28: 34012, TEXTURE29: 34013, TEXTURE30: 34014, TEXTURE31: 34015, ACTIVE_TEXTURE: 34016, REPEAT: 10497, CLAMP_TO_EDGE: 33071, MIRRORED_REPEAT: 33648, FLOAT_VEC2: 35664, FLOAT_VEC3: 35665, FLOAT_VEC4: 35666, INT_VEC2: 35667, INT_VEC3: 35668, INT_VEC4: 35669, BOOL: 35670, BOOL_VEC2: 35671, BOOL_VEC3: 35672, BOOL_VEC4: 35673, FLOAT_MAT2: 35674, FLOAT_MAT3: 35675, FLOAT_MAT4: 35676, SAMPLER_2D: 35678, SAMPLER_CUBE: 35680, VERTEX_ATTRIB_ARRAY_ENABLED: 34338, VERTEX_ATTRIB_ARRAY_SIZE: 34339, VERTEX_ATTRIB_ARRAY_STRIDE: 34340, VERTEX_ATTRIB_ARRAY_TYPE: 34341, VERTEX_ATTRIB_ARRAY_NORMALIZED: 34922, VERTEX_ATTRIB_ARRAY_POINTER: 34373, VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: 34975, COMPILE_STATUS: 35713, LOW_FLOAT: 36336, MEDIUM_FLOAT: 36337, HIGH_FLOAT: 36338, LOW_INT: 36339, MEDIUM_INT: 36340, HIGH_INT: 36341, FRAMEBUFFER: 36160, RENDERBUFFER: 36161, RGBA4: 32854, RGB5_A1: 32855, RGB565: 36194, DEPTH_COMPONENT16: 33189, STENCIL_INDEX: 6401, STENCIL_INDEX8: 36168, DEPTH_STENCIL: 34041, RENDERBUFFER_WIDTH: 36162, RENDERBUFFER_HEIGHT: 36163, RENDERBUFFER_INTERNAL_FORMAT: 36164, RENDERBUFFER_RED_SIZE: 36176, RENDERBUFFER_GREEN_SIZE: 36177, RENDERBUFFER_BLUE_SIZE: 36178, RENDERBUFFER_ALPHA_SIZE: 36179, RENDERBUFFER_DEPTH_SIZE: 36180, RENDERBUFFER_STENCIL_SIZE: 36181, FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE: 36048, FRAMEBUFFER_ATTACHMENT_OBJECT_NAME: 36049, FRAMEBUFFER_ATTACHMENT_TEXTURE_LEVEL: 36050, FRAMEBUFFER_ATTACHMENT_TEXTURE_CUBE_MAP_FACE: 36051, COLOR_ATTACHMENT0: 36064, DEPTH_ATTACHMENT: 36096, STENCIL_ATTACHMENT: 36128, DEPTH_STENCIL_ATTACHMENT: 33306, NONE: 0, FRAMEBUFFER_COMPLETE: 36053, FRAMEBUFFER_INCOMPLETE_ATTACHMENT: 36054, FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: 36055, FRAMEBUFFER_INCOMPLETE_DIMENSIONS: 36057, FRAMEBUFFER_UNSUPPORTED: 36061, FRAMEBUFFER_BINDING: 36006, RENDERBUFFER_BINDING: 36007, MAX_RENDERBUFFER_SIZE: 34024, INVALID_FRAMEBUFFER_OPERATION: 1286, UNPACK_FLIP_Y_WEBGL: 37440, UNPACK_PREMULTIPLY_ALPHA_WEBGL: 37441, CONTEXT_LOST_WEBGL: 37442, UNPACK_COLORSPACE_CONVERSION_WEBGL: 37443, BROWSER_DEFAULT_WEBGL: 37444 } }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(36),
            a = r(46),
            o = r(7),
            s = r(16),
            u = r(25),
            h = r(17),
            l = r(11);
        o.import(r(222));
        var c = new a,
            d = new u({ geometry: c, frustumCulling: !1 }),
            f = new i,
            p = n.extend(function() { return { fragment: "", outputs: null, material: null, blendWithPrevious: !1, clearColor: !1, clearDepth: !0 } }, function() {
                var e = new o({ vertex: o.source("qtek.compositor.vertex"), fragment: this.fragment }),
                    t = new s({ shader: e });
                e.enableTexturesAll(), this.material = t
            }, {
                setUniform: function(e, t) {
                    var r = this.material.uniforms[e];
                    r && (r.value = t)
                },
                getUniform: function(e) { var t = this.material.uniforms[e]; if (t) return t.value },
                attachOutput: function(e, t) { this.outputs || (this.outputs = {}), t = t || l.COLOR_ATTACHMENT0, this.outputs[t] = e },
                detachOutput: function(e) { for (var t in this.outputs) this.outputs[t] === e && (this.outputs[t] = null) },
                bind: function(e, t) {
                    if (this.outputs)
                        for (var r in this.outputs) {
                            var n = this.outputs[r];
                            n && t.attach(n, r)
                        }
                    t && t.bind(e)
                },
                unbind: function(e, t) { t.unbind(e) },
                render: function(e, t) {
                    var r = e.gl;
                    if (t) {
                        this.bind(e, t);
                        var n = h.getExtension(r, "EXT_draw_buffers");
                        if (n && this.outputs) {
                            var i = [];
                            for (var a in this.outputs)(a = +a) >= r.COLOR_ATTACHMENT0 && a <= r.COLOR_ATTACHMENT0 + 8 && i.push(a);
                            n.drawBuffersEXT(i)
                        }
                    }
                    this.trigger("beforerender", this, e);
                    var o = this.clearDepth ? r.DEPTH_BUFFER_BIT : 0;
                    if (r.depthMask(!0), this.clearColor) {
                        o |= r.COLOR_BUFFER_BIT, r.colorMask(!0, !0, !0, !0);
                        var s = this.clearColor;
                        s instanceof Array && r.clearColor(s[0], s[1], s[2], s[3])
                    }
                    r.clear(o), this.blendWithPrevious ? (r.enable(r.BLEND), this.material.transparent = !0) : (r.disable(r.BLEND), this.material.transparent = !1), this.renderQuad(e), this.trigger("afterrender", this, e), t && this.unbind(e, t)
                },
                renderQuad: function(e) { d.material = this.material, e.renderQueue([d], f) },
                dispose: function(e) { this.material.dispose(e) }
            });
        e.exports = p
    }, function(e, t, r) {
        "use strict";

        function n(e) { return "attr_" + e }
        var i = r(196),
            a = r(14),
            o = r(1),
            s = r(20),
            u = r(11),
            h = o.mat4,
            l = o.vec3,
            c = i.StaticAttribute,
            d = l.create,
            f = l.add,
            p = l.set,
            _ = i.extend(function() { return { attributes: { position: new c("position", "float", 3, "POSITION"), texcoord0: new c("texcoord0", "float", 2, "TEXCOORD_0"), texcoord1: new c("texcoord1", "float", 2, "TEXCOORD_1"), normal: new c("normal", "float", 3, "NORMAL"), tangent: new c("tangent", "float", 4, "TANGENT"), color: new c("color", "float", 4, "COLOR"), weight: new c("weight", "float", 3, "WEIGHT"), joint: new c("joint", "float", 4, "JOINT"), barycentric: new c("barycentric", "float", 3, null) }, hint: u.STATIC_DRAW, indices: null, _normalType: "vertex", _enabledAttributes: null } }, {
                updateBoundingBox: function() {
                    var e = this.boundingBox;
                    e || (e = this.boundingBox = new a);
                    var t = this.attributes.position.value;
                    if (t && t.length) {
                        var r = e.min,
                            n = e.max,
                            i = r._array,
                            o = n._array;
                        l.set(i, t[0], t[1], t[2]), l.set(o, t[0], t[1], t[2]);
                        for (var s = 3; s < t.length;) {
                            var u = t[s++],
                                h = t[s++],
                                c = t[s++];
                            u < i[0] && (i[0] = u), h < i[1] && (i[1] = h), c < i[2] && (i[2] = c), u > o[0] && (o[0] = u), h > o[1] && (o[1] = h), c > o[2] && (o[2] = c)
                        }
                        r._dirty = !0, n._dirty = !0
                    }
                },
                dirty: function() {
                    for (var e = this.getEnabledAttributes(), t = 0; t < e.length; t++) this.dirtyAttribute(e[t]);
                    this.dirtyIndices(), this._enabledAttributes = null
                },
                dirtyIndices: function() { this._cache.dirtyAll("indices") },
                dirtyAttribute: function(e) { this._cache.dirtyAll(n(e)), this._cache.dirtyAll("attributes") },
                getTriangleIndices: function(e, t) { if (e < this.triangleCount && e >= 0) { t || (t = d()); var r = this.indices; return t[0] = r[3 * e], t[1] = r[3 * e + 1], t[2] = r[3 * e + 2], t } },
                setTriangleIndices: function(e, t) {
                    var r = this.indices;
                    r[3 * e] = t[0], r[3 * e + 1] = t[1], r[3 * e + 2] = t[2]
                },
                isUseIndices: function() { return !!this.indices },
                initIndicesFromArray: function(e) {
                    var t, r = this.vertexCount > 65535 ? s.Uint32Array : s.Uint16Array;
                    if (e[0] && e[0].length) {
                        var n = 0;
                        t = new r(3 * e.length);
                        for (var i = 0; i < e.length; i++)
                            for (var a = 0; a < 3; a++) t[n++] = e[i][a]
                    } else t = new r(e);
                    this.indices = t
                },
                createAttribute: function(e, t, r, n) { var i = new c(e, t, r, n); return this.attributes[e] && this.removeAttribute(e), this.attributes[e] = i, this._attributeList.push(e), i },
                removeAttribute: function(e) {
                    var t = this._attributeList,
                        r = t.indexOf(e);
                    return r >= 0 && (t.splice(r, 1), delete this.attributes[e], !0)
                },
                getEnabledAttributes: function() {
                    var e = this._enabledAttributes,
                        t = this._attributeList;
                    if (e) return e;
                    for (var r = [], n = this.vertexCount, i = 0; i < t.length; i++) {
                        var a = t[i],
                            o = this.attributes[a];
                        o.value && o.value.length === n * o.size && r.push(a)
                    }
                    return this._enabledAttributes = r, r
                },
                getBufferChunks: function(e) {
                    var t = this._cache;
                    t.use(e.__GLID__);
                    var r = t.isDirty("attributes"),
                        i = t.isDirty("indices");
                    if (r || i) {
                        this._updateBuffer(e, r, i);
                        for (var a = this.getEnabledAttributes(), o = 0; o < a.length; o++) t.fresh(n(a[o]));
                        t.fresh("attributes"), t.fresh("indices")
                    }
                    return t.get("chunks")
                },
                _updateBuffer: function(e, t, r) {
                    var a = this._cache,
                        o = a.get("chunks"),
                        s = !1;
                    o || (o = [], o[0] = { attributeBuffers: [], indicesBuffer: null }, a.put("chunks", o), s = !0);
                    var u = o[0],
                        h = u.attributeBuffers,
                        l = u.indicesBuffer;
                    if (t || s) {
                        var c = this.getEnabledAttributes(),
                            d = {};
                        if (!s)
                            for (var f = 0; f < h.length; f++) d[h[f].name] = h[f];
                        for (var p = 0; p < c.length; p++) {
                            var _, m = c[p],
                                g = this.attributes[m];
                            s || (_ = d[m]);
                            var v;
                            v = _ ? _.buffer : e.createBuffer(), a.isDirty(n(m)) && (e.bindBuffer(e.ARRAY_BUFFER, v), e.bufferData(e.ARRAY_BUFFER, g.value, this.hint)), h[p] = new i.AttributeBuffer(m, g.type, v, g.size, g.semantic)
                        }
                        for (var f = p; f < h.length; f++) e.deleteBuffer(h[f].buffer);
                        h.length = p
                    }
                    this.isUseIndices() && (r || s) && (l || (l = new i.IndicesBuffer(e.createBuffer()), u.indicesBuffer = l), l.count = this.indices.length, e.bindBuffer(e.ELEMENT_ARRAY_BUFFER, l.buffer), e.bufferData(e.ELEMENT_ARRAY_BUFFER, this.indices, this.hint))
                },
                generateVertexNormals: function() {
                    if (this.vertexCount) {
                        var e = this.indices,
                            t = this.attributes,
                            r = t.position.value,
                            n = t.normal.value;
                        if (n && n.length === r.length)
                            for (var i = 0; i < n.length; i++) n[i] = 0;
                        else n = t.normal.value = new s.Float32Array(r.length);
                        for (var a = d(), o = d(), u = d(), h = d(), c = d(), f = d(), _ = 0; _ < e.length;) {
                            var m = e[_++],
                                g = e[_++],
                                v = e[_++];
                            p(a, r[3 * m], r[3 * m + 1], r[3 * m + 2]), p(o, r[3 * g], r[3 * g + 1], r[3 * g + 2]), p(u, r[3 * v], r[3 * v + 1], r[3 * v + 2]), l.sub(h, a, o), l.sub(c, o, u), l.cross(f, h, c);
                            for (var i = 0; i < 3; i++) n[3 * m + i] = n[3 * m + i] + f[i], n[3 * g + i] = n[3 * g + i] + f[i], n[3 * v + i] = n[3 * v + i] + f[i]
                        }
                        for (var i = 0; i < n.length;) p(f, n[i], n[i + 1], n[i + 2]), l.normalize(f, f), n[i++] = f[0], n[i++] = f[1], n[i++] = f[2];
                        this.dirty()
                    }
                },
                generateFaceNormals: function() {
                    if (this.vertexCount) {
                        this.isUniqueVertex() || this.generateUniqueVertex();
                        var e = this.indices,
                            t = this.attributes,
                            r = t.position.value,
                            n = t.normal.value,
                            i = d(),
                            a = d(),
                            o = d(),
                            s = d(),
                            u = d(),
                            h = d();
                        n || (n = t.normal.value = new Float32Array(r.length));
                        for (var c = 0; c < e.length;) {
                            var f = e[c++],
                                _ = e[c++],
                                m = e[c++];
                            p(i, r[3 * f], r[3 * f + 1], r[3 * f + 2]), p(a, r[3 * _], r[3 * _ + 1], r[3 * _ + 2]), p(o, r[3 * m], r[3 * m + 1], r[3 * m + 2]), l.sub(s, i, a), l.sub(u, a, o), l.cross(h, s, u), l.normalize(h, h);
                            for (var g = 0; g < 3; g++) n[3 * f + g] = h[g], n[3 * _ + g] = h[g], n[3 * m + g] = h[g]
                        }
                        this.dirty()
                    }
                },
                generateTangents: function() {
                    if (this.vertexCount) {
                        var e = this.vertexCount,
                            t = this.attributes;
                        t.tangent.value || (t.tangent.value = new Float32Array(4 * e));
                        for (var r = t.texcoord0.value, n = t.position.value, i = t.tangent.value, a = t.normal.value, o = [], s = [], u = 0; u < e; u++) o[u] = [0, 0, 0], s[u] = [0, 0, 0];
                        for (var h = [0, 0, 0], c = [0, 0, 0], p = this.indices, u = 0; u < p.length;) {
                            var _ = p[u++],
                                m = p[u++],
                                g = p[u++],
                                v = r[2 * _],
                                y = r[2 * m],
                                x = r[2 * g],
                                T = r[2 * _ + 1],
                                b = r[2 * m + 1],
                                w = r[2 * g + 1],
                                E = n[3 * _],
                                S = n[3 * m],
                                A = n[3 * g],
                                M = n[3 * _ + 1],
                                N = n[3 * m + 1],
                                C = n[3 * g + 1],
                                L = n[3 * _ + 2],
                                D = n[3 * m + 2],
                                I = n[3 * g + 2],
                                R = S - E,
                                P = A - E,
                                O = N - M,
                                F = C - M,
                                B = D - L,
                                U = I - L,
                                z = y - v,
                                G = x - v,
                                k = b - T,
                                H = w - T,
                                V = 1 / (z * H - k * G);
                            h[0] = (H * R - k * P) * V, h[1] = (H * O - k * F) * V, h[2] = (H * B - k * U) * V, c[0] = (z * P - G * R) * V, c[1] = (z * F - G * O) * V, c[2] = (z * U - G * B) * V, f(o[_], o[_], h), f(o[m], o[m], h), f(o[g], o[g], h), f(s[_], s[_], c), f(s[m], s[m], c), f(s[g], s[g], c)
                        }
                        for (var W = d(), q = d(), X = d(), u = 0; u < e; u++) {
                            X[0] = a[3 * u], X[1] = a[3 * u + 1], X[2] = a[3 * u + 2];
                            var j = o[u];
                            l.scale(W, X, l.dot(X, j)), l.sub(W, j, W), l.normalize(W, W), l.cross(q, X, j), i[4 * u] = W[0], i[4 * u + 1] = W[1], i[4 * u + 2] = W[2], i[4 * u + 3] = l.dot(q, s[u]) < 0 ? -1 : 1
                        }
                        this.dirty()
                    }
                },
                isUniqueVertex: function() { return !this.isUseIndices() || this.vertexCount === this.indices.length },
                generateUniqueVertex: function() {
                    if (this.vertexCount) {
                        this.indices.length > 65535 && (this.indices = new s.Uint32Array(this.indices));
                        for (var e = this.attributes, t = this.indices, r = this.getEnabledAttributes(), n = {}, i = 0; i < r.length; i++) {
                            var a = r[i];
                            n[a] = e[a].value, e[a].init(this.indices.length)
                        }
                        for (var o = 0, u = 0; u < t.length; u++) {
                            for (var h = t[u], i = 0; i < r.length; i++)
                                for (var a = r[i], l = e[a].value, c = e[a].size, d = 0; d < c; d++) l[o * c + d] = n[a][h * c + d];
                            t[u] = o, o++
                        }
                        this.dirty()
                    }
                },
                generateBarycentric: function() {
                    if (this.vertexCount) {
                        this.isUniqueVertex() || this.generateUniqueVertex();
                        var e = this.attributes,
                            t = e.barycentric.value,
                            r = this.indices;
                        if (!t || t.length !== 3 * r.length) {
                            t = e.barycentric.value = new Float32Array(3 * r.length);
                            for (var n = 0; n < r.length;)
                                for (var i = 0; i < 3; i++) {
                                    var a = r[n++];
                                    t[3 * a + i] = 1
                                }
                            this.dirty()
                        }
                    }
                },
                applyTransform: function(e) {
                    var t = this.attributes,
                        r = t.position.value,
                        n = t.normal.value,
                        i = t.tangent.value;
                    e = e._array;
                    var a = h.create();
                    h.invert(a, e), h.transpose(a, a);
                    var o = l.transformMat4,
                        s = l.forEach;
                    s(r, 3, 0, null, o, e), n && s(n, 3, 0, null, o, a), i && s(i, 4, 0, null, o, a), this.boundingBox && this.updateBoundingBox()
                },
                dispose: function(e) {
                    var t = this._cache;
                    t.use(e.__GLID__);
                    var r = t.get("chunks");
                    if (r)
                        for (var n = 0; n < r.length; n++)
                            for (var i = r[n], a = 0; a < i.attributeBuffers.length; a++) {
                                var o = i.attributeBuffers[a];
                                e.deleteBuffer(o.buffer)
                            }
                    t.deleteContext(e.__GLID__)
                }
            });
        Object.defineProperty && (Object.defineProperty(_.prototype, "vertexCount", { enumerable: !1, get: function() { var e = this.attributes[this.mainAttribute]; return e && e.value ? e.value.length / e.size : 0 } }), Object.defineProperty(_.prototype, "triangleCount", { enumerable: !1, get: function() { var e = this.indices; return e ? e.length / 3 : 0 } })), _.Attribute = i.StaticAttribute, e.exports = _
    }, function(e, t, r) {
        "use strict";
        var n = r(3),
            i = r(1),
            a = i.vec3,
            o = a.copy,
            s = a.set,
            u = function(e, t) { this.min = e || new n(1 / 0, 1 / 0, 1 / 0), this.max = t || new n(-1 / 0, -1 / 0, -1 / 0) };
        u.prototype = {
            constructor: u,
            updateFromVertices: function(e) {
                if (e.length > 0) {
                    var t = this.min,
                        r = this.max,
                        n = t._array,
                        i = r._array;
                    o(n, e[0]), o(i, e[0]);
                    for (var a = 1; a < e.length; a++) {
                        var s = e[a];
                        s[0] < n[0] && (n[0] = s[0]), s[1] < n[1] && (n[1] = s[1]), s[2] < n[2] && (n[2] = s[2]), s[0] > i[0] && (i[0] = s[0]), s[1] > i[1] && (i[1] = s[1]), s[2] > i[2] && (i[2] = s[2])
                    }
                    t._dirty = !0, r._dirty = !0
                }
            },
            union: function(e) {
                var t = this.min,
                    r = this.max;
                return a.min(t._array, t._array, e.min._array), a.max(r._array, r._array, e.max._array), t._dirty = !0, r._dirty = !0, this
            },
            intersection: function(e) {
                var t = this.min,
                    r = this.max;
                return a.max(t._array, t._array, e.min._array), a.min(r._array, r._array, e.max._array), t._dirty = !0, r._dirty = !0, this
            },
            intersectBoundingBox: function(e) {
                var t = this.min._array,
                    r = this.max._array,
                    n = e.min._array,
                    i = e.max._array;
                return !(t[0] > i[0] || t[1] > i[1] || t[2] > i[2] || r[0] < n[0] || r[1] < n[1] || r[2] < n[2])
            },
            containBoundingBox: function(e) {
                var t = this.min._array,
                    r = this.max._array,
                    n = e.min._array,
                    i = e.max._array;
                return t[0] <= n[0] && t[1] <= n[1] && t[2] <= n[2] && r[0] >= i[0] && r[1] >= i[1] && r[2] >= i[2]
            },
            containPoint: function(e) {
                var t = this.min._array,
                    r = this.max._array,
                    n = e._array;
                return t[0] <= n[0] && t[1] <= n[1] && t[2] <= n[2] && r[0] >= n[0] && r[1] >= n[1] && r[2] >= n[2]
            },
            isFinite: function() {
                var e = this.min._array,
                    t = this.max._array;
                return isFinite(e[0]) && isFinite(e[1]) && isFinite(e[2]) && isFinite(t[0]) && isFinite(t[1]) && isFinite(t[2])
            },
            applyTransform: function() {
                var e = a.create(),
                    t = a.create(),
                    r = a.create(),
                    n = a.create(),
                    i = a.create(),
                    o = a.create();
                return function(a) {
                    var s = this.min._array,
                        u = this.max._array,
                        h = a._array;
                    return e[0] = h[0] * s[0], e[1] = h[1] * s[0], e[2] = h[2] * s[0], t[0] = h[0] * u[0], t[1] = h[1] * u[0], t[2] = h[2] * u[0], r[0] = h[4] * s[1], r[1] = h[5] * s[1], r[2] = h[6] * s[1], n[0] = h[4] * u[1], n[1] = h[5] * u[1], n[2] = h[6] * u[1], i[0] = h[8] * s[2], i[1] = h[9] * s[2], i[2] = h[10] * s[2], o[0] = h[8] * u[2], o[1] = h[9] * u[2], o[2] = h[10] * u[2], s[0] = Math.min(e[0], t[0]) + Math.min(r[0], n[0]) + Math.min(i[0], o[0]) + h[12], s[1] = Math.min(e[1], t[1]) + Math.min(r[1], n[1]) + Math.min(i[1], o[1]) + h[13], s[2] = Math.min(e[2], t[2]) + Math.min(r[2], n[2]) + Math.min(i[2], o[2]) + h[14], u[0] = Math.max(e[0], t[0]) + Math.max(r[0], n[0]) + Math.max(i[0], o[0]) + h[12], u[1] = Math.max(e[1], t[1]) + Math.max(r[1], n[1]) + Math.max(i[1], o[1]) + h[13], u[2] = Math.max(e[2], t[2]) + Math.max(r[2], n[2]) + Math.max(i[2], o[2]) + h[14], this.min._dirty = !0, this.max._dirty = !0, this
                }
            }(),
            applyProjection: function(e) {
                var t = this.min._array,
                    r = this.max._array,
                    n = e._array,
                    i = t[0],
                    a = t[1],
                    o = t[2],
                    s = r[0],
                    u = r[1],
                    h = t[2],
                    l = r[0],
                    c = r[1],
                    d = r[2];
                if (1 === n[15]) t[0] = n[0] * i + n[12], t[1] = n[5] * a + n[13], r[2] = n[10] * o + n[14], r[0] = n[0] * l + n[12], r[1] = n[5] * c + n[13], t[2] = n[10] * d + n[14];
                else {
                    var f = -1 / o;
                    t[0] = n[0] * i * f, t[1] = n[5] * a * f, r[2] = (n[10] * o + n[14]) * f, f = -1 / h, r[0] = n[0] * s * f, r[1] = n[5] * u * f, f = -1 / d, t[2] = (n[10] * d + n[14]) * f
                }
                return this.min._dirty = !0, this.max._dirty = !0, this
            },
            updateVertices: function() {
                var e = this.vertices;
                if (!e) {
                    for (var e = [], t = 0; t < 8; t++) e[t] = a.fromValues(0, 0, 0);
                    this.vertices = e
                }
                var r = this.min._array,
                    n = this.max._array;
                return s(e[0], r[0], r[1], r[2]), s(e[1], r[0], n[1], r[2]), s(e[2], n[0], r[1], r[2]), s(e[3], n[0], n[1], r[2]), s(e[4], r[0], r[1], n[2]), s(e[5], r[0], n[1], n[2]), s(e[6], n[0], r[1], n[2]), s(e[7], n[0], n[1], n[2]), this
            },
            copy: function(e) {
                var t = this.min,
                    r = this.max;
                return o(t._array, e.min._array), o(r._array, e.max._array), t._dirty = !0, r._dirty = !0, this
            },
            clone: function() { var e = new u; return e.copy(this), e }
        }, e.exports = u
    }, function(e, t) {
        function r(e) {
            if (null == e || "object" != typeof e) return e;
            var t = e,
                n = G.call(e);
            if ("[object Array]" === n) { t = []; for (var i = 0, a = e.length; i < a; i++) t[i] = r(e[i]) } else if (z[n]) {
                var o = e.constructor;
                if (e.constructor.from) t = o.from(e);
                else { t = new o(e.length); for (var i = 0, a = e.length; i < a; i++) t[i] = r(e[i]) }
            } else if (!U[n] && !P(e) && !S(e)) { t = {}; for (var s in e) e.hasOwnProperty(s) && (t[s] = r(e[s])) }
            return t
        }

        function n(e, t, i) {
            if (!w(t) || !w(e)) return i ? r(t) : e;
            for (var a in t)
                if (t.hasOwnProperty(a)) {
                    var o = e[a],
                        s = t[a];
                    !w(s) || !w(o) || x(s) || x(o) || S(s) || S(o) || E(s) || E(o) || P(s) || P(o) ? !i && a in e || (e[a] = r(t[a], !0)) : n(o, s, i)
                }
            return e
        }

        function i(e, t) { for (var r = e[0], i = 1, a = e.length; i < a; i++) r = n(r, e[i], t); return r }

        function a(e, t) { for (var r in t) t.hasOwnProperty(r) && (e[r] = t[r]); return e }

        function o(e, t, r) { for (var n in t) t.hasOwnProperty(n) && (r ? null != t[n] : null == e[n]) && (e[n] = t[n]); return e }

        function s() { return document.createElement("canvas") }

        function u() { return B || (B = Z.createCanvas().getContext("2d")), B }

        function h(e, t) {
            if (e) {
                if (e.indexOf) return e.indexOf(t);
                for (var r = 0, n = e.length; r < n; r++)
                    if (e[r] === t) return r
            }
            return -1
        }

        function l(e, t) {
            function r() {}
            var n = e.prototype;
            r.prototype = t.prototype, e.prototype = new r;
            for (var i in n) e.prototype[i] = n[i];
            e.prototype.constructor = e, e.superClass = t
        }

        function c(e, t, r) { e = "prototype" in e ? e.prototype : e, t = "prototype" in t ? t.prototype : t, o(e, t, r) }

        function d(e) { if (e) return "string" != typeof e && "number" == typeof e.length }

        function f(e, t, r) {
            if (e && t)
                if (e.forEach && e.forEach === H) e.forEach(t, r);
                else if (e.length === +e.length)
                for (var n = 0, i = e.length; n < i; n++) t.call(r, e[n], n, e);
            else
                for (var a in e) e.hasOwnProperty(a) && t.call(r, e[a], a, e)
        }

        function p(e, t, r) { if (e && t) { if (e.map && e.map === q) return e.map(t, r); for (var n = [], i = 0, a = e.length; i < a; i++) n.push(t.call(r, e[i], i, e)); return n } }

        function _(e, t, r, n) { if (e && t) { if (e.reduce && e.reduce === X) return e.reduce(t, r, n); for (var i = 0, a = e.length; i < a; i++) r = t.call(n, r, e[i], i, e); return r } }

        function m(e, t, r) { if (e && t) { if (e.filter && e.filter === V) return e.filter(t, r); for (var n = [], i = 0, a = e.length; i < a; i++) t.call(r, e[i], i, e) && n.push(e[i]); return n } }

        function g(e, t, r) {
            if (e && t)
                for (var n = 0, i = e.length; n < i; n++)
                    if (t.call(r, e[n], n, e)) return e[n]
        }

        function v(e, t) { var r = W.call(arguments, 2); return function() { return e.apply(t, r.concat(W.call(arguments))) } }

        function y(e) { var t = W.call(arguments, 1); return function() { return e.apply(this, t.concat(W.call(arguments))) } }

        function x(e) { return "[object Array]" === G.call(e) }

        function T(e) { return "function" == typeof e }

        function b(e) { return "[object String]" === G.call(e) }

        function w(e) { var t = typeof e; return "function" === t || !!e && "object" == t }

        function E(e) { return !!U[G.call(e)] }

        function S(e) { return "object" == typeof e && "number" == typeof e.nodeType && "object" == typeof e.ownerDocument }

        function A(e) { return e !== e }

        function M(e) {
            for (var t = 0, r = arguments.length; t < r; t++)
                if (null != arguments[t]) return arguments[t]
        }

        function N(e, t) { return null != e ? e : t }

        function C(e, t, r) { return null != e ? e : null != t ? t : r }

        function L() { return Function.call.apply(W, arguments) }

        function D(e) { if ("number" == typeof e) return [e, e, e, e]; var t = e.length; return 2 === t ? [e[0], e[1], e[0], e[1]] : 3 === t ? [e[0], e[1], e[2], e[1]] : e }

        function I(e, t) { if (!e) throw new Error(t) }

        function R(e) { e[j] = !0 }

        function P(e) { return e[j] }

        function O(e) { e && f(e, function(e, t) { this.set(t, e) }, this) }

        function F(e) { return new O(e) }
        var B, U = { "[object Function]": 1, "[object RegExp]": 1, "[object Date]": 1, "[object Error]": 1, "[object CanvasGradient]": 1, "[object CanvasPattern]": 1, "[object Image]": 1, "[object Canvas]": 1 },
            z = { "[object Int8Array]": 1, "[object Uint8Array]": 1, "[object Uint8ClampedArray]": 1, "[object Int16Array]": 1, "[object Uint16Array]": 1, "[object Int32Array]": 1, "[object Uint32Array]": 1, "[object Float32Array]": 1, "[object Float64Array]": 1 },
            G = Object.prototype.toString,
            k = Array.prototype,
            H = k.forEach,
            V = k.filter,
            W = k.slice,
            q = k.map,
            X = k.reduce,
            j = "__ec_primitive__";
        O.prototype = { constructor: O, get: function(e) { return this["_ec_" + e] }, set: function(e, t) { return this["_ec_" + e] = t, t }, each: function(e, t) { void 0 !== t && (e = v(e, t)); for (var r in this) this.hasOwnProperty(r) && e(this[r], r.slice(4)) }, removeKey: function(e) { delete this["_ec_" + e] } };
        var Z = { inherits: l, mixin: c, clone: r, merge: n, mergeAll: i, extend: a, defaults: o, getContext: u, createCanvas: s, indexOf: h, slice: L, find: g, isArrayLike: d, each: f, map: p, reduce: _, filter: m, bind: v, curry: y, isArray: x, isString: b, isObject: w, isFunction: T, isBuiltInObject: E, isDom: S, eqNaN: A, retrieve: M, retrieve2: N, retrieve3: C, assert: I, setAsPrimitive: R, createHashMap: F, normalizeCssArray: D, noop: function() {} };
        e.exports = Z
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(6),
            a = n.extend({ name: "", depthTest: !0, depthMask: !0, transparent: !1, blend: null, _enabledUniforms: null }, function() { this.name || (this.name = "MATERIAL_" + this.__GUID__), this.shader && this.attachShader(this.shader), this.uniforms || (this.uniforms = {}) }, {
                bind: function(e, t, r, n) {
                    for (var t = t || this.shader, a = t.currentTextureSlot(), o = 0; o < this._enabledUniforms.length; o++) {
                        var s = this._enabledUniforms[o],
                            u = this.uniforms[s].value;
                        if (u instanceof i) u.__slot = -1;
                        else if (u instanceof Array)
                            for (var h = 0; h < u.length; h++) u[h] instanceof i && (u[h].__slot = -1)
                    }
                    for (var o = 0; o < this._enabledUniforms.length; o++) {
                        var s = this._enabledUniforms[o],
                            l = this.uniforms[s],
                            u = l.value;
                        if (null !== u)
                            if (u instanceof i)
                                if (u.__slot < 0) {
                                    var c = t.currentTextureSlot(),
                                        d = t.setUniform(e, "1i", s, c);
                                    if (!d) continue;
                                    t.takeCurrentTextureSlot(e, u), u.__slot = c
                                } else t.setUniform(e, "1i", s, u.__slot);
                        else if (u instanceof Array) {
                            if (0 === u.length) continue;
                            var f = u[0];
                            if (f instanceof i) {
                                if (!t.hasUniform(s)) continue;
                                for (var p = [], h = 0; h < u.length; h++) {
                                    var _ = u[h];
                                    if (_.__slot < 0) {
                                        var c = t.currentTextureSlot();
                                        p.push(c), t.takeCurrentTextureSlot(e, _), _.__slot = c
                                    } else p.push(_.__slot)
                                }
                                t.setUniform(e, "1iv", s, p)
                            } else t.setUniform(e, l.type, s, u)
                        } else t.setUniform(e, l.type, s, u);
                        else if ("t" === l.type) {
                            var c = t.currentTextureSlot(),
                                d = t.setUniform(e, "1i", s, c);
                            d && t.takeCurrentTextureSlot(e, null)
                        }
                    }
                    t.resetTextureSlot(a)
                },
                setUniform: function(e, t) {
                    void 0 === t && console.warn('Uniform value "' + e + '" is undefined');
                    var r = this.uniforms[e];
                    r && (r.value = t)
                },
                setUniforms: function(e) {
                    for (var t in e) {
                        var r = e[t];
                        this.setUniform(t, r)
                    }
                },
                isUniformEnabled: function(e) { return this._enabledUniforms.indexOf(e) >= 0 },
                set: function(e, t) {
                    if ("object" == typeof e)
                        for (var r in e) {
                            var n = e[r];
                            this.set(r, n)
                        } else {
                            var i = this.uniforms[e];
                            i && (void 0 === t && (console.warn('Uniform value "' + e + '" is undefined'), t = null), i.value = t)
                        }
                },
                get: function(e) { var t = this.uniforms[e]; if (t) return t.value },
                attachShader: function(e, t) {
                    this.shader && this.shader.detached();
                    var r = this.uniforms;
                    this.uniforms = e.createUniforms(), this.shader = e;
                    var n = this.uniforms;
                    if (this._enabledUniforms = Object.keys(n), this._enabledUniforms.sort(), t)
                        for (var i in r) n[i] && (n[i].value = r[i].value);
                    e.attached()
                },
                detachShader: function() { this.shader.detached(), this.shader = null, this.uniforms = {} },
                clone: function() { var e = new this.constructor({ name: this.name, shader: this.shader }); for (var t in this.uniforms) e.uniforms[t].value = this.uniforms[t].value; return e.depthTest = this.depthTest, e.depthMask = this.depthMask, e.transparent = this.transparent, e.blend = this.blend, e },
                dispose: function(e, t) {
                    if (t)
                        for (var r in this.uniforms) {
                            var n = this.uniforms[r].value;
                            if (n)
                                if (n instanceof i) n.dispose(e);
                                else if (n instanceof Array)
                                for (var a = 0; a < n.length; a++) n[a] instanceof i && n[a].dispose(e)
                        }
                    var o = this.shader;
                    o && (this.detachShader(), o.isAttachedToAny() || o.dispose(e))
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = ["OES_texture_float", "OES_texture_half_float", "OES_texture_float_linear", "OES_texture_half_float_linear", "OES_standard_derivatives", "OES_vertex_array_object", "OES_element_index_uint", "WEBGL_compressed_texture_s3tc", "WEBGL_depth_texture", "EXT_texture_filter_anisotropic", "EXT_shader_texture_lod", "WEBGL_draw_buffers", "EXT_frag_depth", "EXT_sRGB"],
            i = ["MAX_TEXTURE_SIZE", "MAX_CUBE_MAP_TEXTURE_SIZE"],
            a = {},
            o = {},
            s = {
                initialize: function(e) {
                    var t = e.__GLID__;
                    if (!a[t]) {
                        a[t] = {}, o[t] = {};
                        for (var r = 0; r < n.length; r++) {
                            var s = n[r];
                            this._createExtension(e, s)
                        }
                        for (var r = 0; r < i.length; r++) {
                            var u = i[r];
                            o[t][u] = e.getParameter(e[u])
                        }
                    }
                },
                getExtension: function(e, t) { var r = e.__GLID__; if (a[r]) return void 0 === a[r][t] && this._createExtension(e, t), a[r][t] },
                getParameter: function(e, t) { var r = e.__GLID__; if (o[r]) return o[r][t] },
                dispose: function(e) { delete a[e.__GLID__], delete o[e.__GLID__] },
                _createExtension: function(e, t) {
                    var r = e.getExtension(t);
                    r || (r = e.getExtension("MOZ_" + t)), r || (r = e.getExtension("WEBKIT_" + t)), a[e.__GLID__][t] = r
                }
            };
        e.exports = s
    }, function(e, t) {
        e.exports = function(e, t, r) {
            t.eachSeriesByType(e, function(e) {
                var t = e.getData(),
                    r = e.visualColorAccessPath.split(".");
                r[r.length - 1] = "opacity";
                var n = e.get(r);
                t.setVisual("opacity", null == n ? 1 : n), t.hasItemOption && t.each(function(e) {
                    var n = t.getItemModel(e),
                        i = n.get(r);
                    null != i && t.setItemVisual(e, "opacity", i)
                })
            })
        }
    }, function(e, t, r) {
        "use strict";
        var n = r(35),
            i = n.extend(function() { return { color: [1, 1, 1], intensity: 1, castShadow: !0, shadowResolution: 512, group: 0 } }, { type: "", clone: function() { var e = n.prototype.clone.call(this); return e.color = Array.prototype.slice.call(this.color), e.intensity = this.intensity, e.castShadow = this.castShadow, e.shadowResolution = this.shadowResolution, e } });
        e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = !0;
        try { var i = document.createElement("canvas"); if (!(i.getContext("webgl") || i.getContext("experimental-webgl"))) throw new Error } catch (e) { n = !1 }
        var a = {};
        a.supportWebGL = function() { return n }, a.Int8Array = "undefined" == typeof Int8Array ? Array : Int8Array, a.Uint8Array = "undefined" == typeof Uint8Array ? Array : Uint8Array, a.Uint16Array = "undefined" == typeof Uint16Array ? Array : Uint16Array, a.Uint32Array = "undefined" == typeof Uint32Array ? Array : Uint32Array, a.Int16Array = "undefined" == typeof Int16Array ? Array : Int16Array, a.Float32Array = "undefined" == typeof Float32Array ? Array : Float32Array, a.Float64Array = "undefined" == typeof Float64Array ? Array : Float64Array, e.exports = a
    }, function(e, t, r) {
        function n(e) {
            e = e || "perspective", this.layer = null, this.scene = new a, this.rootNode = this.scene, this.viewport = { x: 0, y: 0, width: 0, height: 0 }, this.setProjection(e), this._compositor = new f, this._temporalSS = new p, this._shadowMapPass = new o;
            for (var t = [], r = 0, n = 0; n < 30; n++) {
                for (var i = [], s = 0; s < 6; s++) i.push(4 * _(r, 2) - 2), i.push(4 * _(r, 3) - 2), r++;
                t.push(i)
            }
            this._pcfKernels = t, this.scene.on("beforerender", function(e, t, r) { this.needsTemporalSS() && this._temporalSS.jitterProjection(e, r) }, this)
        }
        var i = r(0),
            a = r(26),
            o = r(211),
            s = r(44),
            u = r(36),
            h = r(9),
            l = r(3),
            c = r(28),
            d = r(53),
            f = r(158),
            p = r(164),
            _ = r(39);
        n.prototype.setProjection = function(e) {
            var t = this.camera;
            t && t.update(), "perspective" === e ? this.camera instanceof s || (this.camera = new s, t && this.camera.setLocalTransform(t.localTransform)) : this.camera instanceof u || (this.camera = new u, t && this.camera.setLocalTransform(t.localTransform)), this.camera.near = .1, this.camera.far = 2e3
        }, n.prototype.setViewport = function(e, t, r, n, i) { this.camera instanceof s && (this.camera.aspect = r / n), i = i || 1, this.viewport.x = e, this.viewport.y = t, this.viewport.width = r, this.viewport.height = n, this.viewport.devicePixelRatio = i, this._compositor.resize(r * i, n * i), this._temporalSS.resize(r * i, n * i) }, n.prototype.containPoint = function(e, t) { var r = this.viewport; return t = this.layer.renderer.getHeight() - t, e >= r.x && t >= r.y && e <= r.x + r.width && t <= r.y + r.height };
        var m = new c;
        n.prototype.castRay = function(e, t, r) {
            var n = this.layer.renderer,
                i = n.viewport;
            return n.viewport = this.viewport, n.screenToNDC(e, t, m), this.camera.castRay(m, r), n.viewport = i, r
        }, n.prototype.prepareRender = function() {
            this.scene.update(), this.camera.update(), this._needsSortProgressively = !1;
            for (var e = 0; e < this.scene.transparentQueue.length; e++) {
                var t = this.scene.transparentQueue[e],
                    r = t.geometry;
                r.needsSortVerticesProgressively && r.needsSortVerticesProgressively() && (this._needsSortProgressively = !0), r.needsSortTrianglesProgressively && r.needsSortTrianglesProgressively() && (this._needsSortProgressively = !0)
            }
            this._frame = 0, this._temporalSS.resetFrame()
        }, n.prototype.render = function(e, t) { this._doRender(e, t, this._frame), this._frame++ }, n.prototype.needsAccumulate = function() { return this.needsTemporalSS() || this._needsSortProgressively }, n.prototype.needsTemporalSS = function() { var e = this._enableTemporalSS; return "auto" == e && (e = this._enablePostEffect), e }, n.prototype.hasDOF = function() { return this._enableDOF }, n.prototype.isAccumulateFinished = function() { return this.needsTemporalSS() ? this._temporalSS.isFinished() : this._frame > 30 }, n.prototype._doRender = function(e, t, r) {
            var n = this.scene,
                i = this.camera;
            if (r = r || 0, this._updateTransparent(e, n, i, r), t || (this._shadowMapPass.kernelPCF = this._pcfKernels[0], this._shadowMapPass.render(e, n, i, !0)), this._updateShadowPCFKernel(r), e.gl.clearColor(0, 0, 0, 0), this._enablePostEffect && (this.needsTemporalSS() && this._temporalSS.jitterProjection(e, i), this._compositor.updateNormal(e, n, i, this._temporalSS.getFrame())), this._updateSSAO(e, n, i, this._temporalSS.getFrame()), this._enablePostEffect) {
                var a = this._compositor.getSourceFrameBuffer();
                a.bind(e), e.gl.clear(e.gl.DEPTH_BUFFER_BIT | e.gl.COLOR_BUFFER_BIT), e.render(n, i, !0, !0), a.unbind(e), this.needsTemporalSS() && t ? (this._compositor.composite(e, i, this._temporalSS.getSourceFrameBuffer(), this._temporalSS.getFrame()), e.setViewport(this.viewport), this._temporalSS.render(e)) : (e.setViewport(this.viewport), this._compositor.composite(e, i, null, 0))
            } else if (this.needsTemporalSS() && t) {
                var a = this._temporalSS.getSourceFrameBuffer();
                a.bind(e), e.saveClear(), e.clearBit = e.gl.DEPTH_BUFFER_BIT | e.gl.COLOR_BUFFER_BIT, e.render(n, i, !0), e.restoreClear(), a.unbind(e), e.setViewport(this.viewport), this._temporalSS.render(e)
            } else e.setViewport(this.viewport), e.render(n, i, !0, !0)
        }, n.prototype._updateTransparent = function(e, t, r, n) {
            for (var i = new l, a = new h, o = r.getWorldPosition(), s = 0; s < t.transparentQueue.length; s++) {
                var u = t.transparentQueue[s],
                    c = u.geometry;
                h.invert(a, u.worldTransform), l.transformMat4(i, o, a), c.needsSortTriangles && c.needsSortTriangles() && c.doSortTriangles(i, n), c.needsSortVertices && c.needsSortVertices() && c.doSortVertices(i, n)
            }
        }, n.prototype._updateSSAO = function(e, t, r, n) {
            var i = this._enableSSAO && this._enablePostEffect;
            i && this._compositor.updateSSAO(e, t, r, this._temporalSS.getFrame());
            for (var a = 0; a < t.opaqueQueue.length; a++) {
                var o = t.opaqueQueue[a];
                o.renderNormal && o.material.shader[i ? "enableTexture" : "disableTexture"]("ssaoMap"), i && o.material.set("ssaoMap", this._compositor.getSSAOTexture())
            }
        }, n.prototype._updateShadowPCFKernel = function(e) { for (var t = this._pcfKernels[e % this._pcfKernels.length], r = this.scene.opaqueQueue, n = 0; n < r.length; n++) r[n].receiveShadow && (r[n].material.set("pcfKernel", t), r[n].material.shader.define("fragment", "PCF_KERNEL_SIZE", t.length / 2)) }, n.prototype.dispose = function(e) { this._compositor.dispose(e.gl), this._temporalSS.dispose(e.gl), this._shadowMapPass.dispose(e) }, n.prototype.setPostEffect = function(e, t) {
            var r = this._compositor;
            this._enablePostEffect = e.get("enable");
            var n = e.getModel("bloom"),
                i = e.getModel("edge"),
                a = e.getModel("DOF", e.getModel("depthOfField")),
                o = e.getModel("SSAO", e.getModel("screenSpaceAmbientOcclusion")),
                s = e.getModel("SSR", e.getModel("screenSpaceReflection")),
                u = e.getModel("FXAA"),
                h = e.getModel("colorCorrection");
            n.get("enable") ? r.enableBloom() : r.disableBloom(), a.get("enable") ? r.enableDOF() : r.disableDOF(), s.get("enable") ? r.enableSSR() : r.disableSSR(), h.get("enable") ? r.enableColorCorrection() : r.disableColorCorrection(), i.get("enable") ? r.enableEdge() : r.disableEdge(), u.get("enable") ? r.enableFXAA() : r.disableFXAA(), this._enableDOF = a.get("enable"), this._enableSSAO = o.get("enable"), this._enableSSAO ? r.enableSSAO() : r.disableSSAO(), r.setBloomIntensity(n.get("intensity")), r.setEdgeColor(i.get("color")), r.setColorLookupTexture(h.get("lookupTexture"), t), r.setExposure(h.get("exposure")), ["radius", "quality", "intensity"].forEach(function(e) { r.setSSAOParameter(e, o.get(e)) }), ["quality", "maxRoughness"].forEach(function(e) { r.setSSRParameter(e, s.get(e)) }), ["quality", "focalDistance", "focalRange", "blurRadius", "fstop"].forEach(function(e) { r.setDOFParameter(e, a.get(e)) }), ["brightness", "contrast", "saturation"].forEach(function(e) { r.setColorCorrection(e, h.get(e)) })
        }, n.prototype.setDOFFocusOnPoint = function(e) { if (this._enablePostEffect) { if (e > this.camera.far || e < this.camera.near) return; return this._compositor.setDOFParameter("focalDistance", e), !0 } }, n.prototype.setTemporalSuperSampling = function(e) { this._enableTemporalSS = e.get("enable") }, n.prototype.isLinearSpace = function() { return this._enablePostEffect }, n.prototype.setRootNode = function(e) {
            if (this.rootNode !== e) {
                for (var t = this.rootNode.children(), r = 0; r < t.length; r++) e.add(t[r]);
                e !== this.scene && this.scene.add(e), this.rootNode = e
            }
        }, n.prototype.add = function(e) { this.rootNode.add(e) }, n.prototype.remove = function(e) { this.rootNode.remove(e) }, n.prototype.removeAll = function(e) { this.rootNode.removeAll(e) }, i.util.extend(n.prototype, d), e.exports = n
    }, function(e, t, r) {
        var n = r(13),
            i = r(1).vec3,
            a = r(0),
            o = r(34),
            s = [
                [0, 0],
                [1, 1]
            ],
            u = n.extend(function() { return { segmentScale: 1, dynamic: !0, useNativeLine: !0, attributes: { position: new n.Attribute("position", "float", 3, "POSITION"), positionPrev: new n.Attribute("positionPrev", "float", 3), positionNext: new n.Attribute("positionNext", "float", 3), prevPositionPrev: new n.Attribute("prevPositionPrev", "float", 3), prevPosition: new n.Attribute("prevPosition", "float", 3), prevPositionNext: new n.Attribute("prevPositionNext", "float", 3), offset: new n.Attribute("offset", "float", 1), color: new n.Attribute("color", "float", 4, "COLOR") } } }, {
                resetOffset: function() { this._vertexOffset = 0, this._triangleOffset = 0, this._itemVertexOffsets = [] },
                setVertexCount: function(e) {
                    var t = this.attributes;
                    this.vertexCount !== e && (t.position.init(e), t.color.init(e), this.useNativeLine || (t.positionPrev.init(e), t.positionNext.init(e), t.offset.init(e)), e > 65535 ? this.indices instanceof Uint16Array && (this.indices = new Uint32Array(this.indices)) : this.indices instanceof Uint32Array && (this.indices = new Uint16Array(this.indices)))
                },
                setTriangleCount: function(e) { this.triangleCount !== e && (this.indices = 0 === e ? null : this.vertexCount > 65535 ? new Uint32Array(3 * e) : new Uint16Array(3 * e)) },
                _getCubicCurveApproxStep: function(e, t, r, n) { return 1 / (i.dist(e, t) + i.dist(r, t) + i.dist(n, r) + 1) * this.segmentScale },
                getCubicCurveVertexCount: function(e, t, r, n) {
                    var i = this._getCubicCurveApproxStep(e, t, r, n),
                        a = Math.ceil(1 / i);
                    return this.useNativeLine ? 2 * a : 2 * a + 2
                },
                getCubicCurveTriangleCount: function(e, t, r, n) {
                    var i = this._getCubicCurveApproxStep(e, t, r, n),
                        a = Math.ceil(1 / i);
                    return this.useNativeLine ? 0 : 2 * a
                },
                getLineVertexCount: function() { return this.getPolylineVertexCount(s) },
                getLineTriangleCount: function() { return this.getPolylineTriangleCount(s) },
                getPolylineVertexCount: function(e) {
                    var t = "number" != typeof e[0],
                        r = t ? e.length : e.length / 3;
                    return this.useNativeLine ? 2 * (r - 1) : 2 * (r - 1) + 2
                },
                getPolylineTriangleCount: function(e) {
                    var t = "number" != typeof e[0],
                        r = t ? e.length : e.length / 3;
                    return this.useNativeLine ? 0 : 2 * Math.max(r - 1, 0)
                },
                addCubicCurve: function(e, t, r, n, i, a) { null == a && (a = 1); for (var o = e[0], s = e[1], u = e[2], h = t[0], l = t[1], c = t[2], d = r[0], f = r[1], p = r[2], _ = n[0], m = n[1], g = n[2], v = this._getCubicCurveApproxStep(e, t, r, n), y = v * v, x = y * v, T = 3 * v, b = 3 * y, w = 6 * y, E = 6 * x, S = o - 2 * h + d, A = s - 2 * l + f, M = u - 2 * c + p, N = 3 * (h - d) - o + _, C = 3 * (l - f) - s + m, L = 3 * (c - p) - u + g, D = o, I = s, R = u, P = (h - o) * T + S * b + N * x, O = (l - s) * T + A * b + C * x, F = (c - u) * T + M * b + L * x, B = S * w + N * E, U = A * w + C * E, z = M * w + L * E, G = N * E, k = C * E, H = L * E, V = 0, W = 0, q = Math.ceil(1 / v), X = new Float32Array(3 * (q + 1)), X = [], j = 0, W = 0; W < q + 1; W++) X[j++] = D, X[j++] = I, X[j++] = R, D += P, I += O, R += F, P += B, O += U, F += z, B += G, U += k, z += H, (V += v) > 1 && (D = P > 0 ? Math.min(D, _) : Math.max(D, _), I = O > 0 ? Math.min(I, m) : Math.max(I, m), R = F > 0 ? Math.min(R, g) : Math.max(R, g)); return this.addPolyline(X, i, a, !1) },
                addLine: function(e, t, r, n) { return this.addPolyline([e, t], r, n, !1) },
                addPolyline: function(e, t, r, n) {
                    if (e.length) {
                        this._itemVertexOffsets.push(this._vertexOffset);
                        var i, a, o = "number" != typeof e[0],
                            s = this.attributes.position,
                            u = this.attributes.positionPrev,
                            h = this.attributes.positionNext,
                            l = this.attributes.color,
                            c = this.attributes.offset,
                            d = this.indices,
                            f = this._vertexOffset,
                            p = o ? e.length : e.length / 3,
                            _ = p;
                        if (!(p < 2)) {
                            null == r && (r = 1), r = Math.max(r, .01);
                            for (var m = 0; m < _; m++) {
                                if (o) i = e[m], a = n ? t[m] : t;
                                else {
                                    var g = 3 * m;
                                    if (i = i || [], i[0] = e[g], i[1] = e[g + 1], i[2] = e[g + 2], n) {
                                        var v = 4 * m;
                                        a = a || [], a[0] = t[v], a[1] = t[v + 1], a[2] = t[v + 2], a[3] = t[v + 3]
                                    } else a = t
                                }
                                if (this.useNativeLine ? m > 1 && (s.copy(f, f - 1), l.copy(f, f - 1), f++) : (m < _ - 1 && (u.set(f + 2, i), u.set(f + 3, i)), m > 0 && (h.set(f - 2, i), h.set(f - 1, i)), s.set(f, i), s.set(f + 1, i), l.set(f, a), l.set(f + 1, a), c.set(f, r / 2), c.set(f + 1, -r / 2), f += 2), this.useNativeLine) l.set(f, a), s.set(f, i), f++;
                                else if (m > 0) {
                                    var y = 3 * this._triangleOffset,
                                        d = this.indices;
                                    d[y] = f - 4, d[y + 1] = f - 3, d[y + 2] = f - 2, d[y + 3] = f - 3, d[y + 4] = f - 1, d[y + 5] = f - 2, this._triangleOffset += 2
                                }
                            }
                            if (!this.useNativeLine) {
                                var x = this._vertexOffset,
                                    T = this._vertexOffset + 2 * p;
                                u.copy(x, x + 2), u.copy(x + 1, x + 3), h.copy(T - 1, T - 3), h.copy(T - 2, T - 4)
                            }
                            return this._vertexOffset = f, this._vertexOffset
                        }
                    }
                },
                setItemColor: function(e, t) {
                    for (var r = this._itemVertexOffsets[e], n = e < this._itemVertexOffsets.length - 1 ? this._itemVertexOffsets[e + 1] : this._vertexOffset, i = r; i < n; i++) this.attributes.color.set(i, t);
                    this.dirty("color")
                },
                currentTriangleOffset: function() { return this._triangleOffset },
                currentVertexOffset: function() { return this._vertexOffset }
            });
        a.util.defaults(u.prototype, o), e.exports = u
    }, function(e, t, r) {
        function n(e) { return "CANVAS" === e.nodeName || "VIDEO" === e.nodeName || e.complete }
        var i = r(6),
            a = r(17),
            o = r(11),
            s = r(27),
            u = r(80),
            h = u.isPowerOfTwo,
            l = ["px", "nx", "py", "ny", "pz", "nz"],
            c = i.extend(function() { return { image: { px: null, nx: null, py: null, ny: null, pz: null, nz: null }, pixels: { px: null, nx: null, py: null, ny: null, pz: null, nz: null }, mipmaps: [] } }, {
                update: function(e) {
                    e.bindTexture(e.TEXTURE_CUBE_MAP, this._cache.get("webgl_texture")), this.updateCommon(e);
                    var t = this.format,
                        r = this.type;
                    e.texParameteri(e.TEXTURE_CUBE_MAP, e.TEXTURE_WRAP_S, this.wrapS), e.texParameteri(e.TEXTURE_CUBE_MAP, e.TEXTURE_WRAP_T, this.wrapT), e.texParameteri(e.TEXTURE_CUBE_MAP, e.TEXTURE_MAG_FILTER, this.magFilter), e.texParameteri(e.TEXTURE_CUBE_MAP, e.TEXTURE_MIN_FILTER, this.minFilter);
                    var n = a.getExtension(e, "EXT_texture_filter_anisotropic");
                    if (n && this.anisotropic > 1 && e.texParameterf(e.TEXTURE_CUBE_MAP, n.TEXTURE_MAX_ANISOTROPY_EXT, this.anisotropic), 36193 === r) { a.getExtension(e, "OES_texture_half_float") || (r = o.FLOAT) }
                    if (this.mipmaps.length)
                        for (var i = this.width, s = this.height, u = 0; u < this.mipmaps.length; u++) {
                            var h = this.mipmaps[u];
                            this._updateTextureData(e, h, u, i, s, t, r), i /= 2, s /= 2
                        } else this._updateTextureData(e, this, 0, this.width, this.height, t, r), !this.NPOT && this.useMipmap && e.generateMipmap(e.TEXTURE_CUBE_MAP);
                    e.bindTexture(e.TEXTURE_CUBE_MAP, null)
                },
                _updateTextureData: function(e, t, r, n, i, a, o) {
                    for (var s = 0; s < 6; s++) {
                        var u = l[s],
                            h = t.image && t.image[u];
                        h ? e.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + s, r, a, a, o, h) : e.texImage2D(e.TEXTURE_CUBE_MAP_POSITIVE_X + s, r, a, n, i, 0, a, o, t.pixels && t.pixels[u])
                    }
                },
                generateMipmap: function(e) { this.useMipmap && !this.NPOT && (e.bindTexture(e.TEXTURE_CUBE_MAP, this._cache.get("webgl_texture")), e.generateMipmap(e.TEXTURE_CUBE_MAP)) },
                bind: function(e) { e.bindTexture(e.TEXTURE_CUBE_MAP, this.getWebGLTexture(e)) },
                unbind: function(e) { e.bindTexture(e.TEXTURE_CUBE_MAP, null) },
                isPowerOfTwo: function() { return this.image.px ? h(this.image.px.width) && h(this.image.px.height) : h(this.width) && h(this.height) },
                isRenderable: function() { return this.image.px ? n(this.image.px) && n(this.image.nx) && n(this.image.py) && n(this.image.ny) && n(this.image.pz) && n(this.image.nz) : !(!this.width || !this.height) },
                load: function(e, t) {
                    var r = 0,
                        n = this;
                    return s.each(e, function(e, i) {
                        var a = new Image;
                        t && (a.crossOrigin = t), a.onload = function() { r--, 0 === r && (n.dirty(), n.trigger("success", n)), a.onload = null }, a.onerror = function() { r--, a.onerror = null }, r++, a.src = e, n.image[i] = a
                    }), this
                }
            });
        Object.defineProperty(c.prototype, "width", { get: function() { return this.image && this.image.px ? this.image.px.width : this._width }, set: function(e) { this.image && this.image.px ? console.warn("Texture from image can't set width") : (this._width !== e && this.dirty(), this._width = e) } }), Object.defineProperty(c.prototype, "height", { get: function() { return this.image && this.image.px ? this.image.px.height : this._height }, set: function(e) { this.image && this.image.px ? console.warn("Texture from image can't set height") : (this._height !== e && this.dirty(), this._height = e) } }), e.exports = c
    }, function(e, t, r) {
        var n = r(0),
            i = {};
        i.getFormattedLabel = function(e, t, r, i, a) {
            r = r || "normal";
            var o = e.getData(i),
                s = o.getItemModel(t),
                u = e.getDataParams(t, i);
            null != a && u.value instanceof Array && (u.value = u.value[a]);
            var h = s.get("normal" === r ? ["label", "formatter"] : ["emphasis", "label", "formatter"]);
            null == h && (h = s.get(["label", "formatter"]));
            var l;
            return "function" == typeof h ? (u.status = r, l = h(u)) : "string" == typeof h && (l = n.format.formatTpl(h, u)), l
        }, i.normalizeToArray = function(e) { return e instanceof Array ? e : null == e ? [] : [e] }, e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = r(70),
            i = r(11),
            a = r(5),
            o = n.extend({ skeleton: null, joints: null, useSkinMatricesTexture: !1 }, function() { this.joints || (this.joints = []) }, {
                render: function(e, t) {
                    if (t = t || this.material.shader, this.skeleton) {
                        this.skeleton.update();
                        var r = this.skeleton.getSubSkinMatrices(this.__GUID__, this.joints);
                        if (this.useSkinMatricesTexture) {
                            var i, a = this.joints.length;
                            i = a > 256 ? 64 : a > 64 ? 32 : a > 16 ? 16 : 8;
                            var o = this.getSkinMatricesTexture();
                            o.width = i, o.height = i, o.pixels && o.pixels.length === i * i * 4 || (o.pixels = new Float32Array(i * i * 4)), o.pixels.set(r), o.dirty(), t.setUniform(e, "1f", "skinMatricesTextureSize", i)
                        } else t.setUniformOfSemantic(e, "SKIN_MATRIX", r)
                    }
                    return n.prototype.render.call(this, e, t)
                },
                getSkinMatricesTexture: function() { return this._skinMatricesTexture = this._skinMatricesTexture || new a({ type: i.FLOAT, minFilter: i.NEAREST, magFilter: i.NEAREST, useMipmap: !1, flipY: !1 }), this._skinMatricesTexture }
            });
        o.POINTS = i.POINTS, o.LINES = i.LINES, o.LINE_LOOP = i.LINE_LOOP, o.LINE_STRIP = i.LINE_STRIP, o.TRIANGLES = i.TRIANGLES, o.TRIANGLE_STRIP = i.TRIANGLE_STRIP, o.TRIANGLE_FAN = i.TRIANGLE_FAN, o.BACK = i.BACK, o.FRONT = i.FRONT, o.FRONT_AND_BACK = i.FRONT_AND_BACK, o.CW = i.CW, o.CCW = i.CCW, e.exports = o
    }, function(e, t, r) {
        "use strict";

        function n(e, t) { if (t.castShadow && !e.castShadow) return !0 }
        var i = r(35),
            a = r(19),
            o = r(14),
            s = i.extend(function() { return { material: null, autoUpdate: !0, opaqueQueue: [], transparentQueue: [], lights: [], viewBoundingBoxLastFrame: new o, _lightUniforms: {}, _lightNumber: {}, _opaqueObjectCount: 0, _transparentObjectCount: 0, _nodeRepository: {} } }, function() { this._scene = this }, {
                addToScene: function(e) { e.name && (this._nodeRepository[e.name] = e) },
                removeFromScene: function(e) { e.name && delete this._nodeRepository[e.name] },
                getNode: function(e) { return this._nodeRepository[e] },
                cloneNode: function(e) {
                    var t = e.clone(),
                        r = {},
                        n = function(i, a) { i.skeleton && (a.skeleton = i.skeleton.clone(e, t), a.joints = i.joints.slice()), i.material && (r[i.material.__GUID__] = { oldMat: i.material }); for (var o = 0; o < i._children.length; o++) n(i._children[o], a._children[o]) };
                    n(e, t);
                    for (var i in r) r[i].newMat = r[i].oldMat.clone();
                    return t.traverse(function(e) { e.material && (e.material = r[e.material.__GUID__].newMat) }), t
                },
                update: function(e, t) {
                    if (this.autoUpdate || e) {
                        i.prototype.update.call(this, e);
                        var r = this.lights,
                            n = this.material && this.material.transparent;
                        if (this._opaqueObjectCount = 0, this._transparentObjectCount = 0, r.length = 0, this._updateRenderQueue(this, n), this.opaqueQueue.length = this._opaqueObjectCount, this.transparentQueue.length = this._transparentObjectCount, !t) {
                            var a = this._lightNumber;
                            for (var o in a)
                                for (var s in a[o]) a[o][s] = 0;
                            for (var u = 0; u < r.length; u++) {
                                var h = r[u],
                                    o = h.group;
                                a[o] || (a[o] = {}), a[o][h.type] = a[o][h.type] || 0, a[o][h.type]++
                            }
                            this._updateLightUniforms()
                        }
                    }
                },
                _updateRenderQueue: function(e, t) {
                    if (!e.invisible)
                        for (var r = 0; r < e._children.length; r++) {
                            var n = e._children[r];
                            n instanceof a && this.lights.push(n), n.isRenderable() && (n.material.transparent || t ? this.transparentQueue[this._transparentObjectCount++] = n : this.opaqueQueue[this._opaqueObjectCount++] = n), n._children.length > 0 && this._updateRenderQueue(n)
                        }
                },
                _updateLightUniforms: function() {
                    var e = this.lights;
                    e.sort(n);
                    var t = this._lightUniforms;
                    for (var r in t)
                        for (var i in t[r]) t[r][i].value.length = 0;
                    for (var a = 0; a < e.length; a++) {
                        var o = e[a],
                            r = o.group;
                        for (var i in o.uniformTemplates) {
                            var s = o.uniformTemplates[i];
                            t[r] || (t[r] = {}), t[r][i] || (t[r][i] = { type: "", value: [] });
                            var u = s.value(o),
                                h = t[r][i];
                            switch (h.type = s.type + "v", s.type) {
                                case "1i":
                                case "1f":
                                case "t":
                                    h.value.push(u);
                                    break;
                                case "2f":
                                case "3f":
                                case "4f":
                                    for (var l = 0; l < u.length; l++) h.value.push(u[l]);
                                    break;
                                default:
                                    console.error("Unkown light uniform type " + s.type)
                            }
                        }
                    }
                },
                isShaderLightNumberChanged: function(e) {
                    var t = e.lightGroup;
                    for (var r in this._lightNumber[t])
                        if (this._lightNumber[t][r] !== e.lightNumber[r]) return !0;
                    for (var r in e.lightNumber)
                        if (this._lightNumber[t][r] !== e.lightNumber[r]) return !0;
                    return !1
                },
                setShaderLightNumber: function(e) {
                    var t = e.lightGroup;
                    for (var r in this._lightNumber[t]) e.lightNumber[r] = this._lightNumber[t][r];
                    e.dirty()
                },
                setLightUniforms: function(e, t) {
                    var r = e.lightGroup;
                    for (var n in this._lightUniforms[r]) {
                        var i = this._lightUniforms[r][n];
                        if ("tv" === i.type)
                            for (var a = 0; a < i.value.length; a++) {
                                var o = i.value[a],
                                    s = e.currentTextureSlot(),
                                    u = e.setUniform(t, "1i", n, s);
                                u && e.takeCurrentTextureSlot(t, o)
                            } else e.setUniform(t, i.type, n, i.value)
                    }
                },
                dispose: function() { this.material = null, this.opaqueQueue = [], this.transparentQueue = [], this.lights = [], this._lightUniforms = {}, this._lightNumber = {}, this._nodeRepository = {} }
            });
        e.exports = s
    }, function(e, t, r) {
        "use strict";
        var n = 0,
            i = Array.prototype,
            a = i.forEach,
            o = {
                genGUID: function() { return ++n },
                relative2absolute: function(e, t) {
                    if (!t || e.match(/^\//)) return e;
                    for (var r = e.split("/"), n = t.split("/"), i = r[0];
                        "." === i || ".." === i;) ".." === i && n.pop(), r.shift(), i = r[0];
                    return n.join("/") + "/" + r.join("/")
                },
                extend: function(e, t) {
                    if (t)
                        for (var r in t) t.hasOwnProperty(r) && (e[r] = t[r]);
                    return e
                },
                defaults: function(e, t) {
                    if (t)
                        for (var r in t) void 0 === e[r] && (e[r] = t[r]);
                    return e
                },
                extendWithPropList: function(e, t, r) {
                    if (t)
                        for (var n = 0; n < r.length; n++) {
                            var i = r[n];
                            e[i] = t[i]
                        }
                    return e
                },
                defaultsWithPropList: function(e, t, r) {
                    if (t)
                        for (var n = 0; n < r.length; n++) {
                            var i = r[n];
                            null == e[i] && (e[i] = t[i])
                        }
                    return e
                },
                each: function(e, t, r) {
                    if (e && t)
                        if (e.forEach && e.forEach === a) e.forEach(t, r);
                        else if (e.length === +e.length)
                        for (var n = 0, i = e.length; n < i; n++) t.call(r, e[n], n, e);
                    else
                        for (var o in e) e.hasOwnProperty(o) && t.call(r, e[o], o, e)
                },
                isObject: function(e) { return e === Object(e) },
                isArray: function(e) { return e instanceof Array },
                isArrayLike: function(e) { return !!e && e.length === +e.length },
                clone: function(e) { if (o.isObject(e)) { if (o.isArray(e)) return e.slice(); if (o.isArrayLike(e)) { for (var t = new e.constructor(e.length), r = 0; r < e.length; r++) t[r] = e[r]; return t } return o.extend({}, e) } return e }
            };
        e.exports = o
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.vec2,
            a = function(e, t) { e = e || 0, t = t || 0, this._array = i.fromValues(e, t), this._dirty = !0 };
        if (a.prototype = { constructor: a, add: function(e) { return i.add(this._array, this._array, e._array), this._dirty = !0, this }, set: function(e, t) { return this._array[0] = e, this._array[1] = t, this._dirty = !0, this }, setArray: function(e) { return this._array[0] = e[0], this._array[1] = e[1], this._dirty = !0, this }, clone: function() { return new a(this.x, this.y) }, copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this }, cross: function(e, t) { return i.cross(e._array, this._array, t._array), e._dirty = !0, this }, dist: function(e) { return i.dist(this._array, e._array) }, distance: function(e) { return i.distance(this._array, e._array) }, div: function(e) { return i.div(this._array, this._array, e._array), this._dirty = !0, this }, divide: function(e) { return i.divide(this._array, this._array, e._array), this._dirty = !0, this }, dot: function(e) { return i.dot(this._array, e._array) }, len: function() { return i.len(this._array) }, length: function() { return i.length(this._array) }, lerp: function(e, t, r) { return i.lerp(this._array, e._array, t._array, r), this._dirty = !0, this }, min: function(e) { return i.min(this._array, this._array, e._array), this._dirty = !0, this }, max: function(e) { return i.max(this._array, this._array, e._array), this._dirty = !0, this }, mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this }, multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this }, negate: function() { return i.negate(this._array, this._array), this._dirty = !0, this }, normalize: function() { return i.normalize(this._array, this._array), this._dirty = !0, this }, random: function(e) { return i.random(this._array, e), this._dirty = !0, this }, scale: function(e) { return i.scale(this._array, this._array, e), this._dirty = !0, this }, scaleAndAdd: function(e, t) { return i.scaleAndAdd(this._array, this._array, e._array, t), this._dirty = !0, this }, sqrDist: function(e) { return i.sqrDist(this._array, e._array) }, squaredDistance: function(e) { return i.squaredDistance(this._array, e._array) }, sqrLen: function() { return i.sqrLen(this._array) }, squaredLength: function() { return i.squaredLength(this._array) }, sub: function(e) { return i.sub(this._array, this._array, e._array), this._dirty = !0, this }, subtract: function(e) { return i.subtract(this._array, this._array, e._array), this._dirty = !0, this }, transformMat2: function(e) { return i.transformMat2(this._array, this._array, e._array), this._dirty = !0, this }, transformMat2d: function(e) { return i.transformMat2d(this._array, this._array, e._array), this._dirty = !0, this }, transformMat3: function(e) { return i.transformMat3(this._array, this._array, e._array), this._dirty = !0, this }, transformMat4: function(e) { return i.transformMat4(this._array, this._array, e._array), this._dirty = !0, this }, toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" }, toArray: function() { return Array.prototype.slice.call(this._array) } }, Object.defineProperty) {
            var o = a.prototype;
            Object.defineProperty(o, "x", { get: function() { return this._array[0] }, set: function(e) { this._array[0] = e, this._dirty = !0 } }), Object.defineProperty(o, "y", { get: function() { return this._array[1] }, set: function(e) { this._array[1] = e, this._dirty = !0 } })
        }
        a.add = function(e, t, r) { return i.add(e._array, t._array, r._array), e._dirty = !0, e }, a.set = function(e, t, r) { return i.set(e._array, t, r), e._dirty = !0, e }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.cross = function(e, t, r) { return i.cross(e._array, t._array, r._array), e._dirty = !0, e }, a.dist = function(e, t) { return i.distance(e._array, t._array) }, a.distance = a.dist, a.div = function(e, t, r) { return i.divide(e._array, t._array, r._array), e._dirty = !0, e }, a.divide = a.div, a.dot = function(e, t) { return i.dot(e._array, t._array) }, a.len = function(e) { return i.length(e._array) }, a.lerp = function(e, t, r, n) { return i.lerp(e._array, t._array, r._array, n), e._dirty = !0, e }, a.min = function(e, t, r) { return i.min(e._array, t._array, r._array), e._dirty = !0, e }, a.max = function(e, t, r) { return i.max(e._array, t._array, r._array), e._dirty = !0, e }, a.mul = function(e, t, r) { return i.multiply(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.negate = function(e, t) { return i.negate(e._array, t._array), e._dirty = !0, e }, a.normalize = function(e, t) { return i.normalize(e._array, t._array), e._dirty = !0, e }, a.random = function(e, t) { return i.random(e._array, t), e._dirty = !0, e }, a.scale = function(e, t, r) { return i.scale(e._array, t._array, r), e._dirty = !0, e }, a.scaleAndAdd = function(e, t, r, n) { return i.scaleAndAdd(e._array, t._array, r._array, n), e._dirty = !0, e }, a.sqrDist = function(e, t) { return i.sqrDist(e._array, t._array) }, a.squaredDistance = a.sqrDist, a.sqrLen = function(e) { return i.sqrLen(e._array) }, a.squaredLength = a.sqrLen, a.sub = function(e, t, r) { return i.subtract(e._array, t._array, r._array), e._dirty = !0, e }, a.subtract = a.sub, a.transformMat2 = function(e, t, r) { return i.transformMat2(e._array, t._array, r._array), e._dirty = !0, e }, a.transformMat2d = function(e, t, r) { return i.transformMat2d(e._array, t._array, r._array), e._dirty = !0, e }, a.transformMat3 = function(e, t, r) { return i.transformMat3(e._array, t._array, r._array), e._dirty = !0, e }, a.transformMat4 = function(e, t, r) { return i.transformMat4(e._array, t._array, r._array), e._dirty = !0, e }, e.exports = a
    }, function(e, t, r) {
        function n(e, t) {
            var r = [];
            return i.util.each(e.dimensions, function(n) {
                var i = e.getDimensionInfo(n),
                    a = i.otherDims,
                    o = a[t];
                null != o && !1 !== o && (r[o] = i.name)
            }), r
        }
        var i = r(0);
        e.exports = function(e, t, r) {
            var a = e.getData(),
                o = e.getRawValue(t),
                s = i.util.isArray(o) ? function(e) {
                    function o(e, t) {
                        var n = a.getDimensionInfo(t);
                        if (n && !1 !== n.otherDims.tooltip) {
                            var o = n.type,
                                h = (s ? "- " + (n.tooltipName || n.name) + ": " : "") + ("ordinal" === o ? e + "" : "time" === o ? r ? "" : i.format.formatTime("yyyy/MM/dd hh:mm:ss", e) : i.format.addCommas(e));
                            h && u.push(i.format.encodeHTML(h))
                        }
                    }
                    var s = !0,
                        u = [],
                        h = n(a, "tooltip");
                    return h.length ? i.util.each(h, function(e) { o(a.get(e, t), e) }) : i.util.each(e, o), (s ? "<br/>" : "") + u.join(s ? "<br/>" : ", ")
                }(o) : i.format.encodeHTML(i.format.addCommas(o)),
                u = a.getName(t),
                h = a.getItemVisual(t, "color");
            i.util.isObject(h) && h.colorStops && (h = (h.colorStops[0] || {}).color), h = h || "transparent";
            var l = i.format.getTooltipMarker(h),
                c = e.name;
            return "\0-" === c && (c = ""), c = c ? i.format.encodeHTML(c) + (r ? ": " : "<br/>") : "", r ? l + c + s : c + l + (u ? i.format.encodeHTML(u) + ": " + s : s)
        }
    }, function(e, t, r) {
        function n() {}
        var i = r(2),
            a = r(57),
            o = r(58),
            s = r(0);
        n.prototype = {
            constructor: n,
            setScene: function(e) { this._scene = e, this._skybox && this._skybox.attachScene(this._scene) },
            initLight: function(e) { this._lightRoot = e, this.mainLight = new i.DirectionalLight({ shadowBias: .005 }), this.ambientLight = new i.AmbientLight, e.add(this.mainLight), e.add(this.ambientLight) },
            dispose: function() { this._lightRoot && (this._lightRoot.remove(this.mainLight), this._lightRoot.remove(this.ambientLight)) },
            updateLight: function(e) {
                var t = this.mainLight,
                    r = this.ambientLight,
                    n = e.getModel("light"),
                    a = n.getModel("main"),
                    o = n.getModel("ambient");
                t.intensity = a.get("intensity"), r.intensity = o.get("intensity"), t.color = i.parseColor(a.get("color")).slice(0, 3), r.color = i.parseColor(o.get("color")).slice(0, 3);
                var s = a.get("alpha") || 0,
                    u = a.get("beta") || 0;
                t.position.setArray(i.directionFromAlphaBeta(s, u)), t.lookAt(i.Vector3.ZERO), t.castShadow = a.get("shadow"), t.shadowResolution = i.getShadowResolution(a.get("shadowQuality"))
            },
            updateAmbientCubemap: function(e, t, r) {
                var n = t.getModel("light.ambientCubemap"),
                    o = n.get("texture");
                if (o) {
                    this._cubemapLightsCache = this._cubemapLightsCache || {};
                    var s = this._cubemapLightsCache[o];
                    if (!s) {
                        var u = this;
                        s = this._cubemapLightsCache[o] = i.createAmbientCubemap(n.option, e, r, function() { u._skybox instanceof a && u._skybox.setEnvironmentMap(s.specular.cubemap), r.getZr().refresh() })
                    }
                    this._lightRoot.add(s.diffuse), this._lightRoot.add(s.specular), this._currentCubemapLights = s
                } else this._currentCubemapLights && (this._lightRoot.remove(this._currentCubemapLights.diffuse), this._lightRoot.remove(this._currentCubemapLights.specular), this._currentCubemapLights = null)
            },
            updateSkybox: function(e, t, r) {
                function n() { return h._skybox instanceof o || (h._skybox && h._skybox.dispose(e.gl), h._skybox = new o), h._skybox }
                var u = t.get("environment"),
                    h = this;
                if (u && "none" !== u)
                    if ("auto" === u)
                        if (this._currentCubemapLights) {
                            var l = function() { return h._skybox instanceof a || (h._skybox && h._skybox.dispose(e.gl), h._skybox = new a), h._skybox }(),
                                c = this._currentCubemapLights.specular.cubemap;
                            l.setEnvironmentMap(c), this._scene && l.attachScene(this._scene), l.material.set("lod", 2)
                        } else this._skybox && this._skybox.detachScene();
                else if ("object" == typeof u && u.colorStops || "string" == typeof u && s.color.parse(u)) {
                    var d = n(),
                        f = new i.Texture2D({ anisotropic: 8, flipY: !1 });
                    d.setEnvironmentMap(f);
                    var p = f.image = document.createElement("canvas");
                    p.width = p.height = 16;
                    var _ = p.getContext("2d"),
                        m = new s.graphic.Rect({ shape: { x: 0, y: 0, width: 16, height: 16 }, style: { fill: u } });
                    m.brush(_), d.attachScene(this._scene)
                } else {
                    var d = n(),
                        f = i.loadTexture(u, r, { anisotropic: 8, flipY: !1 });
                    d.setEnvironmentMap(f), d.attachScene(this._scene)
                } else this._skybox && this._skybox.detachScene(this._scene), this._skybox = null;
                var g = t.coordinateSystem;
                if (this._skybox)
                    if (!g || !g.viewGL || "auto" === u || u.match && u.match(/.hdr$/)) this._skybox.material.shader.undefine("fragment", "SRGB_DECODE");
                    else {
                        var v = g.viewGL.isLinearSpace() ? "define" : "undefine";
                        this._skybox.material.shader[v]("fragment", "SRGB_DECODE")
                    }
            }
        }, e.exports = n
    }, function(e, t) { e.exports = { defaultOption: { light: { main: { shadow: !1, shadowQuality: "high", color: "#fff", intensity: 1, alpha: 0, beta: 0 }, ambient: { color: "#fff", intensity: .2 }, ambientCubemap: { texture: null, exposure: 1, diffuseIntensity: .5, specularIntensity: .5 } } } } }, function(e, t) { e.exports = { defaultOption: { postEffect: { enable: !1, bloom: { enable: !0, intensity: .1 }, depthOfField: { enable: !1, focalRange: 20, focalDistance: 50, blurRadius: 10, fstop: 2.8, quality: "medium" }, screenSpaceAmbientOcclusion: { enable: !1, radius: 2, quality: "medium", intensity: 1 }, screenSpaceReflection: { enable: !1, quality: "medium", maxRoughness: .8 }, colorCorrection: { enable: !0, exposure: 0, brightness: 0, contrast: 1, saturation: 1, lookupTexture: "" }, edge: { enable: !1 }, FXAA: { enable: !1 } }, temporalSuperSampling: { enable: "auto" } } } }, function(e, t) { e.exports = { defaultOption: { shading: null, realisticMaterial: { textureTiling: 1, textureOffset: 0, detailTexture: null }, lambertMaterial: { textureTiling: 1, textureOffset: 0, detailTexture: null }, colorMaterial: { textureTiling: 1, textureOffset: 0, detailTexture: null }, hatchingMaterial: { textureTiling: 1, textureOffset: 0, paperColor: "#fff" } } } }, function(e, t) {
        e.exports = {
            convertToDynamicArray: function(e) {
                e && this.resetOffset();
                var t = this.attributes;
                for (var r in t) e || !t[r].value ? t[r].value = [] : t[r].value = Array.prototype.slice.call(t[r].value);
                e || !this.indices ? this.indices = [] : this.indices = Array.prototype.slice.call(this.indices)
            },
            convertToTypedArray: function() {
                var e = this.attributes;
                for (var t in e) e[t].value && e[t].value.length > 0 ? e[t].value = new Float32Array(e[t].value) : e[t].value = null;
                this.indices && this.indices.length > 0 && (this.indices = this.vertexCount > 65535 ? new Uint32Array(this.indices) : new Uint16Array(this.indices)), this.dirty()
            }
        }
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(3),
            a = r(55),
            o = r(9),
            s = r(1),
            u = r(14),
            h = s.mat4,
            l = 0,
            c = n.extend({ name: "", position: null, rotation: null, scale: null, worldTransform: null, localTransform: null, autoUpdateLocalTransform: !0, _parent: null, _scene: null, _needsUpdateWorldTransform: !0, _inIterating: !1, __depth: 0 }, function() { this.name || (this.name = (this.type || "NODE") + "_" + l++), this.position || (this.position = new i), this.rotation || (this.rotation = new a), this.scale || (this.scale = new i(1, 1, 1)), this.worldTransform = new o, this.localTransform = new o, this._children = [] }, {
                target: null,
                invisible: !1,
                isRenderable: function() { return !1 },
                setName: function(e) {
                    var t = this._scene;
                    if (t) {
                        var r = t._nodeRepository;
                        delete r[this.name], r[e] = this
                    }
                    this.name = e
                },
                add: function(e) {
                    this._inIterating && console.warn("Add operation can cause unpredictable error when in iterating");
                    var t = e._parent;
                    if (t !== this) {
                        t && t.remove(e), e._parent = this, this._children.push(e);
                        var r = this._scene;
                        r && r !== e.scene && e.traverse(this._addSelfToScene, this), e._needsUpdateWorldTransform = !0
                    }
                },
                remove: function(e) {
                    this._inIterating && console.warn("Remove operation can cause unpredictable error when in iterating");
                    var t = this._children,
                        r = t.indexOf(e);
                    r < 0 || (t.splice(r, 1), e._parent = null, this._scene && e.traverse(this._removeSelfFromScene, this))
                },
                removeAll: function() {
                    for (var e = this._children, t = 0; t < e.length; t++) e[t]._parent = null, this._scene && e[t].traverse(this._removeSelfFromScene, this);
                    this._children = []
                },
                getScene: function() { return this._scene },
                getParent: function() { return this._parent },
                _removeSelfFromScene: function(e) { e._scene.removeFromScene(e), e._scene = null },
                _addSelfToScene: function(e) { this._scene.addToScene(e), e._scene = this._scene },
                isAncestor: function(e) {
                    for (var t = e._parent; t;) {
                        if (t === this) return !0;
                        t = t._parent
                    }
                    return !1
                },
                children: function() { return this._children.slice() },
                childAt: function(e) { return this._children[e] },
                getChildByName: function(e) {
                    for (var t = this._children, r = 0; r < t.length; r++)
                        if (t[r].name === e) return t[r]
                },
                getDescendantByName: function(e) { for (var t = this._children, r = 0; r < t.length; r++) { var n = t[r]; if (n.name === e) return n; var i = n.getDescendantByName(e); if (i) return i } },
                queryNode: function(e) { if (e) { for (var t = e.split("/"), r = this, n = 0; n < t.length; n++) { var i = t[n]; if (i) { for (var a = !1, o = r._children, s = 0; s < o.length; s++) { var u = o[s]; if (u.name === i) { r = u, a = !0; break } } if (!a) return } } return r } },
                getPath: function(e) { if (!this._parent) return "/"; for (var t = this._parent, r = this.name; t._parent && (r = t.name + "/" + r, t._parent != e);) t = t._parent; return !t._parent && e ? null : r },
                traverse: function(e, t, r) {
                    this._inIterating = !0, null != r && this.constructor !== r || e.call(t, this);
                    for (var n = this._children, i = 0, a = n.length; i < a; i++) n[i].traverse(e, t, r);
                    this._inIterating = !1
                },
                eachChild: function(e, t, r) {
                    this._inIterating = !0;
                    for (var n = this._children, i = null == r, a = 0, o = n.length; a < o; a++) {
                        var s = n[a];
                        (i || s.constructor === r) && e.call(t, s, a)
                    }
                    this._inIterating = !1
                },
                setLocalTransform: function(e) { h.copy(this.localTransform._array, e._array), this.decomposeLocalTransform() },
                decomposeLocalTransform: function(e) {
                    var t = e ? null : this.scale;
                    this.localTransform.decomposeMatrix(t, this.rotation, this.position)
                },
                setWorldTransform: function(e) { h.copy(this.worldTransform._array, e._array), this.decomposeWorldTransform() },
                decomposeWorldTransform: function() {
                    var e = h.create();
                    return function(t) {
                        var r = this.localTransform,
                            n = this.worldTransform;
                        this._parent ? (h.invert(e, this._parent.worldTransform._array), h.multiply(r._array, e, n._array)) : h.copy(r._array, n._array);
                        var i = t ? null : this.scale;
                        r.decomposeMatrix(i, this.rotation, this.position)
                    }
                }(),
                transformNeedsUpdate: function() { return this.position._dirty || this.rotation._dirty || this.scale._dirty },
                updateLocalTransform: function() {
                    var e = this.position,
                        t = this.rotation,
                        r = this.scale;
                    if (this.transformNeedsUpdate()) {
                        var n = this.localTransform._array;
                        h.fromRotationTranslation(n, t._array, e._array), h.scale(n, n, r._array), t._dirty = !1, r._dirty = !1, e._dirty = !1, this._needsUpdateWorldTransform = !0
                    }
                },
                _updateWorldTransformTopDown: function() {
                    var e = this.localTransform._array,
                        t = this.worldTransform._array;
                    this._parent ? h.multiplyAffine(t, this._parent.worldTransform._array, e) : h.copy(t, e)
                },
                updateWorldTransform: function() {
                    for (var e = this; e && e.getParent() && e.getParent().transformNeedsUpdate();) e = e.getParent();
                    e.update()
                },
                update: function(e) { this.autoUpdateLocalTransform ? this.updateLocalTransform() : e = !0, (e || this._needsUpdateWorldTransform) && (this._updateWorldTransformTopDown(), e = !0, this._needsUpdateWorldTransform = !1); for (var t = this._children, r = 0, n = t.length; r < n; r++) t[r].update(e) },
                getBoundingBox: function() {
                    function e(e) { return !e.invisible }
                    return function(t, r) {
                        r = r || new u, t = t || e;
                        var n = this._children;
                        0 === n.length && (r.max.set(-1 / 0, -1 / 0, -1 / 0), r.min.set(1 / 0, 1 / 0, 1 / 0));
                        for (var i = new u, a = 0; a < n.length; a++) {
                            var o = n[a];
                            t(o) && (o.getBoundingBox(t, i), o.updateLocalTransform(), i.isFinite() && i.applyTransform(o.localTransform), 0 === a ? r.copy(i) : r.union(i))
                        }
                        return r
                    }
                }(),
                getWorldPosition: function(e) { this.transformNeedsUpdate() && this.updateWorldTransform(); var t = this.worldTransform._array; if (e) { var r = e._array; return r[0] = t[12], r[1] = t[13], r[2] = t[14], e } return new i(t[12], t[13], t[14]) },
                clone: function() {
                    var e = new this.constructor,
                        t = this._children;
                    e.setName(this.name), e.position.copy(this.position), e.rotation.copy(this.rotation), e.scale.copy(this.scale);
                    for (var r = 0; r < t.length; r++) e.add(t[r].clone());
                    return e
                },
                rotateAround: function() {
                    var e = new i,
                        t = new o;
                    return function(r, n, i) {
                        e.copy(this.position).subtract(r);
                        var a = this.localTransform;
                        a.identity(), a.translate(r), a.rotate(i, n), t.fromRotationTranslation(this.rotation, e), a.multiply(t), a.scale(this.scale), this.decomposeLocalTransform(), this._needsUpdateWorldTransform = !0
                    }
                }(),
                lookAt: function() { var e = new o; return function(t, r) { e.lookAt(this.position, t, r || this.localTransform.y).invert(), this.setLocalTransform(e), this.target = t } }()
            });
        e.exports = c
    }, function(e, t, r) {
        "use strict";
        var n = r(69),
            i = n.extend({ left: -1, right: 1, near: -1, far: 1, top: 1, bottom: -1 }, {
                updateProjectionMatrix: function() { this.projectionMatrix.ortho(this.left, this.right, this.bottom, this.top, this.near, this.far) },
                decomposeProjectionMatrix: function() {
                    var e = this.projectionMatrix._array;
                    this.left = (-1 - e[12]) / e[0], this.right = (1 - e[12]) / e[0], this.top = (1 - e[13]) / e[5], this.bottom = (-1 - e[13]) / e[5], this.near = -(-1 - e[14]) / e[10], this.far = -(1 - e[14]) / e[10]
                },
                clone: function() { var e = n.prototype.clone.call(this); return e.left = this.left, e.right = this.right, e.near = this.near, e.far = this.far, e.top = this.top, e.bottom = this.bottom, e }
            });
        e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = n.extend(function() { return { name: "", inputLinks: {}, outputLinks: {}, _prevOutputTextures: {}, _outputTextures: {}, _outputReferences: {}, _rendering: !1, _rendered: !1, _compositor: null } }, {
                updateParameter: function(e, t) {
                    var r = this.outputs[e],
                        n = r.parameters,
                        i = r._parametersCopy;
                    if (i || (i = r._parametersCopy = {}), n)
                        for (var a in n) "width" !== a && "height" !== a && (i[a] = n[a]);
                    var o, s;
                    return o = n.width instanceof Function ? n.width.call(this, t) : n.width, s = n.height instanceof Function ? n.height.call(this, t) : n.height, i.width === o && i.height === s || this._outputTextures[e] && this._outputTextures[e].dispose(t.gl), i.width = o, i.height = s, i
                },
                setParameter: function(e, t) {},
                getParameter: function(e) {},
                setParameters: function(e) { for (var t in e) this.setParameter(t, e[t]) },
                render: function() {},
                getOutput: function(e, t) { if (null == t) return t = e, this._outputTextures[t]; var r = this.outputs[t]; if (r) return this._rendered ? r.outputLastFrame ? this._prevOutputTextures[t] : this._outputTextures[t] : this._rendering ? (this._prevOutputTextures[t] || (this._prevOutputTextures[t] = this._compositor.allocateTexture(r.parameters || {})), this._prevOutputTextures[t]) : (this.render(e), this._outputTextures[t]) },
                removeReference: function(e) { if (0 === --this._outputReferences[e]) { this.outputs[e].keepLastFrame ? (this._prevOutputTextures[e] && this._compositor.releaseTexture(this._prevOutputTextures[e]), this._prevOutputTextures[e] = this._outputTextures[e]) : this._compositor.releaseTexture(this._outputTextures[e]) } },
                link: function(e, t, r) { this.inputLinks[e] = { node: t, pin: r }, t.outputLinks[r] || (t.outputLinks[r] = []), t.outputLinks[r].push({ node: this, pin: e }), this.pass.material.shader.enableTexture(e) },
                clear: function() { this.inputLinks = {}, this.outputLinks = {} },
                updateReference: function(e) {
                    if (!this._rendering) {
                        this._rendering = !0;
                        for (var t in this.inputLinks) {
                            var r = this.inputLinks[t];
                            r.node.updateReference(r.pin)
                        }
                        this._rendering = !1
                    }
                    e && this._outputReferences[e]++
                },
                beforeFrame: function() { this._rendered = !1; for (var e in this.outputLinks) this._outputReferences[e] = 0 },
                afterFrame: function() {
                    for (var e in this.outputLinks)
                        if (this._outputReferences[e] > 0) {
                            var t = this.outputs[e];
                            t.keepLastFrame ? (this._prevOutputTextures[e] && this._compositor.releaseTexture(this._prevOutputTextures[e]), this._prevOutputTextures[e] = this._outputTextures[e]) : this._compositor.releaseTexture(this._outputTextures[e])
                        }
                }
            });
        e.exports = i
    }, function(e, t) { e.exports = { defaultOption: { viewControl: { projection: "perspective", autoRotate: !1, autoRotateDirection: "cw", autoRotateSpeed: 10, autoRotateAfterStill: 3, damping: .8, rotateSensitivity: 1, zoomSensitivity: 1, panSensitivity: 1, panMouseButton: "middle", rotateMouseButton: "left", distance: 150, minDistance: 40, maxDistance: 400, orthographicSize: 150, maxOrthographicSize: 400, minOrthographicSize: 20, center: [0, 0, 0], alpha: 0, beta: 0, minAlpha: -90, maxAlpha: 90 } }, setView: function(e) { e = e || {}, this.option.viewControl = this.option.viewControl || {}, null != e.alpha && (this.option.viewControl.alpha = e.alpha), null != e.beta && (this.option.viewControl.beta = e.beta), null != e.distance && (this.option.viewControl.distance = e.distance), null != e.center && (this.option.viewControl.center = e.center) } } }, function(e, t) {
        function r(e, t) { for (var r = 0, n = 1 / t, i = e; i > 0;) r += n * (i % t), i = Math.floor(i / t), n /= t; return r }
        e.exports = r
    }, function(e, t, r) {
        function n(e) { return e instanceof Array || (e = [e, e]), e }
        var i = r(8),
            a = r(28),
            o = r(3),
            s = (r(55), r(4)),
            u = s.firstNotNull,
            h = { left: 0, middle: 1, right: 2 },
            l = i.extend(function() { return { zr: null, viewGL: null, _center: new o, minDistance: .5, maxDistance: 1.5, maxOrthographicSize: 300, minOrthographicSize: 30, minAlpha: -90, maxAlpha: 90, minBeta: -1 / 0, maxBeta: 1 / 0, autoRotateAfterStill: 0, autoRotateDirection: "cw", autoRotateSpeed: 60, damping: .8, rotateSensitivity: 1, zoomSensitivity: 1, panSensitivity: 1, panMouseButton: "middle", rotateMouseButton: "left", _mode: "rotate", _camera: null, _needsUpdate: !1, _rotating: !1, _phi: 0, _theta: 0, _mouseX: 0, _mouseY: 0, _rotateVelocity: new a, _panVelocity: new a, _distance: 500, _zoomSpeed: 0, _stillTimeout: 0, _animators: [] } }, function() {
                ["_mouseDownHandler", "_mouseWheelHandler", "_mouseMoveHandler", "_mouseUpHandler", "_pinchHandler", "_contextMenuHandler", "_update"].forEach(function(e) { this[e] = this[e].bind(this) }, this)
            }, {
                init: function() {
                    var e = this.zr;
                    e && (e.on("mousedown", this._mouseDownHandler), e.on("globalout", this._mouseUpHandler), e.on("mousewheel", this._mouseWheelHandler), e.on("pinch", this._pinchHandler), e.animation.on("frame", this._update), e.dom.addEventListener("contextmenu", this._contextMenuHandler))
                },
                dispose: function() {
                    var e = this.zr;
                    e && (e.off("mousedown", this._mouseDownHandler), e.off("mousemove", this._mouseMoveHandler), e.off("mouseup", this._mouseUpHandler), e.off("mousewheel", this._mouseWheelHandler), e.off("pinch", this._pinchHandler), e.off("globalout", this._mouseUpHandler), e.dom.removeEventListener("contextmenu", this._contextMenuHandler), e.animation.off("frame", this._update)), this.stopAllAnimation()
                },
                getDistance: function() { return this._distance },
                setDistance: function(e) { this._distance = e, this._needsUpdate = !0 },
                getOrthographicSize: function() { return this._orthoSize },
                setOrthographicSize: function(e) { this._orthoSize = e, this._needsUpdate = !0 },
                getAlpha: function() { return this._theta / Math.PI * 180 },
                getBeta: function() { return -this._phi / Math.PI * 180 },
                getCenter: function() { return this._center.toArray() },
                setAlpha: function(e) { e = Math.max(Math.min(this.maxAlpha, e), this.minAlpha), this._theta = e / 180 * Math.PI, this._needsUpdate = !0 },
                setBeta: function(e) { e = Math.max(Math.min(this.maxBeta, e), this.minBeta), this._phi = -e / 180 * Math.PI, this._needsUpdate = !0 },
                setCenter: function(e) { this._center.setArray(e) },
                setViewGL: function(e) { this.viewGL = e },
                getCamera: function() { return this.viewGL.camera },
                setFromViewControlModel: function(e, t) {
                    t = t || {};
                    var r = t.baseDistance || 0,
                        n = t.baseOrthoSize || 1,
                        i = e.get("projection");
                    "perspective" !== i && "orthographic" !== i && "isometric" !== i && (i = "perspective"), this._projection = i, this.viewGL.setProjection(i);
                    var a = e.get("distance") + r,
                        o = e.get("orthographicSize") + n;
                    [
                        ["damping", .8],
                        ["autoRotate", !1],
                        ["autoRotateAfterStill", 3],
                        ["autoRotateDirection", "cw"],
                        ["autoRotateSpeed", 10],
                        ["minDistance", 30],
                        ["maxDistance", 400],
                        ["minOrthographicSize", 30],
                        ["maxOrthographicSize", 300],
                        ["minAlpha", -90],
                        ["maxAlpha", 90],
                        ["minBeta", -1 / 0],
                        ["maxBeta", 1 / 0],
                        ["rotateSensitivity", 1],
                        ["zoomSensitivity", 1],
                        ["panSensitivity", 1],
                        ["panMouseButton", "left"],
                        ["rotateMouseButton", "middle"]
                    ].forEach(function(t) { this[t[0]] = u(e.get(t[0]), t[1]) }, this), this.minDistance += r, this.maxDistance += r, this.minOrthographicSize += n, this.maxOrthographicSize += n;
                    var s = e.ecModel,
                        h = {};
                    ["animation", "animationDurationUpdate", "animationEasingUpdate"].forEach(function(t) { h[t] = u(e.get(t), s && s.get(t)) });
                    var l = u(t.alpha, e.get("alpha")) || 0,
                        c = u(t.beta, e.get("beta")) || 0,
                        d = u(t.center, e.get("center")) || [0, 0, 0];
                    h.animation && h.animationDurationUpdate > 0 && this._notFirst ? this.animateTo({ alpha: l, beta: c, center: d, distance: a, targetOrthographicSize: o, easing: h.animationEasingUpdate, duration: h.animationDurationUpdate }) : (this.setDistance(a), this.setAlpha(l), this.setBeta(c), this.setCenter(d), this.setOrthographicSize(o)), this._notFirst = !0, this._validateProperties()
                },
                _validateProperties: function() {},
                animateTo: function(e) {
                    var t = this.zr,
                        r = this,
                        n = {},
                        i = {};
                    return null != e.distance && (n.distance = this.getDistance(), i.distance = e.distance), null != e.orthographicSize && (n.orthographicSize = this.getOrthographicSize(), i.orthographicSize = e.orthographicSize), null != e.alpha && (n.alpha = this.getAlpha(), i.alpha = e.alpha), null != e.beta && (n.beta = this.getBeta(), i.beta = e.beta), null != e.center && (n.center = this.getCenter(), i.center = e.center), this._addAnimator(t.animation.animate(n).when(e.duration || 1e3, i).during(function() { null != n.alpha && r.setAlpha(n.alpha), null != n.beta && r.setBeta(n.beta), null != n.distance && r.setDistance(n.distance), null != n.center && r.setCenter(n.center), null != n.orthographicSize && r.setOrthographicSize(n.orthographicSize), r._needsUpdate = !0 })).start(e.easing || "linear")
                },
                stopAllAnimation: function() {
                    for (var e = 0; e < this._animators.length; e++) this._animators[e].stop();
                    this._animators.length = 0
                },
                _isAnimating: function() { return this._animators.length > 0 },
                _update: function(e) {
                    if (this._rotating) {
                        var t = ("cw" === this.autoRotateDirection ? 1 : -1) * this.autoRotateSpeed / 180 * Math.PI;
                        this._phi -= t * e / 1e3, this._needsUpdate = !0
                    } else this._rotateVelocity.len() > 0 && (this._needsUpdate = !0);
                    (Math.abs(this._zoomSpeed) > .1 || this._panVelocity.len() > 0) && (this._needsUpdate = !0), this._needsUpdate && (e = Math.min(e, 50), this._updateDistanceOrSize(e), this._updatePan(e), this._updateRotate(e), this._updateTransform(), this.getCamera().update(), this.zr && this.zr.refresh(), this.trigger("update"), this._needsUpdate = !1)
                },
                _updateRotate: function(e) {
                    var t = this._rotateVelocity;
                    this._phi = t.y * e / 20 + this._phi, this._theta = t.x * e / 20 + this._theta, this.setAlpha(this.getAlpha()), this.setBeta(this.getBeta()), this._vectorDamping(t, Math.pow(this.damping, e / 16))
                },
                _updateDistanceOrSize: function(e) { "perspective" === this._projection ? this._setDistance(this._distance + this._zoomSpeed * e / 20) : this._setOrthoSize(this._orthoSize + this._zoomSpeed * e / 20), this._zoomSpeed *= Math.pow(this.damping, e / 16) },
                _setDistance: function(e) { this._distance = Math.max(Math.min(e, this.maxDistance), this.minDistance) },
                _setOrthoSize: function(e) {
                    this._orthoSize = Math.max(Math.min(e, this.maxOrthographicSize), this.minOrthographicSize);
                    var t = this.getCamera(),
                        r = this._orthoSize,
                        n = r / this.viewGL.viewport.height * this.viewGL.viewport.width;
                    t.left = -n / 2, t.right = n / 2, t.top = r / 2, t.bottom = -r / 2
                },
                _updatePan: function(e) {
                    var t = this._panVelocity,
                        r = this._distance,
                        n = this.getCamera(),
                        i = n.worldTransform.y,
                        a = n.worldTransform.x;
                    this._center.scaleAndAdd(a, -t.x * r / 200).scaleAndAdd(i, -t.y * r / 200), this._vectorDamping(t, 0)
                },
                _updateTransform: function() {
                    var e = this.getCamera(),
                        t = new o,
                        r = this._theta + Math.PI / 2,
                        n = this._phi + Math.PI / 2,
                        i = Math.sin(r);
                    t.x = i * Math.cos(n), t.y = -Math.cos(r), t.z = i * Math.sin(n), e.position.copy(this._center).scaleAndAdd(t, this._distance), e.rotation.identity().rotateY(-this._phi).rotateX(-this._theta)
                },
                _startCountingStill: function() {
                    clearTimeout(this._stillTimeout);
                    var e = this.autoRotateAfterStill,
                        t = this;
                    !isNaN(e) && e > 0 && (this._stillTimeout = setTimeout(function() { t._rotating = !0 }, 1e3 * e))
                },
                _vectorDamping: function(e, t) {
                    var r = e.len();
                    r *= t, r < 1e-4 && (r = 0), e.normalize().scale(r)
                },
                _decomposeTransform: function() {
                    if (this.getCamera()) {
                        var e = new o;
                        e.eulerFromQuat(this.getCamera().rotation.normalize(), "ZYX"), this._theta = -e.x, this._phi = -e.y, this.setBeta(this.getBeta()), this.setAlpha(this.getAlpha()), this.getCamera().aspect ? this._setDistance(this.getCamera().position.dist(this._center)) : this._setOrthoSize(this.getCamera().top - this.getCamera().bottom)
                    }
                },
                _mouseDownHandler: function(e) {
                    if (!e.target && !this._isAnimating()) {
                        var t = e.offsetX,
                            r = e.offsetY;
                        this.viewGL && !this.viewGL.containPoint(t, r) || (this.zr.on("mousemove", this._mouseMoveHandler), this.zr.on("mouseup", this._mouseUpHandler), e.event.targetTouches ? 1 === e.event.targetTouches.length && (this._mode = "rotate") : e.event.button === h[this.rotateMouseButton] ? this._mode = "rotate" : e.event.button === h[this.panMouseButton] ? this._mode = "pan" : this._mode = "", this._rotateVelocity.set(0, 0), this._rotating = !1, this.autoRotate && this._startCountingStill(), this._mouseX = e.offsetX, this._mouseY = e.offsetY)
                    }
                },
                _mouseMoveHandler: function(e) {
                    if (!(e.target && e.target.__isGLToZRProxy || this._isAnimating())) {
                        var t = n(this.panSensitivity),
                            r = n(this.rotateSensitivity);
                        "rotate" === this._mode ? (this._rotateVelocity.y = (e.offsetX - this._mouseX) / this.zr.getHeight() * 2 * r[0], this._rotateVelocity.x = (e.offsetY - this._mouseY) / this.zr.getWidth() * 2 * r[1]) : "pan" === this._mode && (this._panVelocity.x = (e.offsetX - this._mouseX) / this.zr.getWidth() * t[0] * 400, this._panVelocity.y = (-e.offsetY + this._mouseY) / this.zr.getHeight() * t[1] * 400), this._mouseX = e.offsetX, this._mouseY = e.offsetY, e.event.preventDefault()
                    }
                },
                _mouseWheelHandler: function(e) {
                    if (!this._isAnimating()) {
                        var t = e.event.wheelDelta || -e.event.detail;
                        this._zoomHandler(e, t)
                    }
                },
                _pinchHandler: function(e) { this._isAnimating() || (this._zoomHandler(e, e.pinchScale > 1 ? 1 : -1), this._mode = "") },
                _zoomHandler: function(e, t) {
                    if (0 !== t) {
                        var r = e.offsetX,
                            n = e.offsetY;
                        if (!this.viewGL || this.viewGL.containPoint(r, n)) {
                            var i;
                            i = "perspective" === this._projection ? Math.max(Math.max(Math.min(this._distance - this.minDistance, this.maxDistance - this._distance)) / 20, .5) : Math.max(Math.max(Math.min(this._orthoSize - this.minOrthographicSize, this.maxOrthographicSize - this._orthoSize)) / 20, .5), this._zoomSpeed = (t > 0 ? -1 : 1) * i * this.zoomSensitivity, this._rotating = !1, this.autoRotate && "rotate" === this._mode && this._startCountingStill(), e.event.preventDefault()
                        }
                    }
                },
                _mouseUpHandler: function() { this.zr.off("mousemove", this._mouseMoveHandler), this.zr.off("mouseup", this._mouseUpHandler) },
                _isRightMouseButtonUsed: function() { return "right" === this.rotateMouseButton || "right" === this.panMouseButton },
                _contextMenuHandler: function(e) { this._isRightMouseButtonUsed() && e.preventDefault() },
                _addAnimator: function(e) {
                    var t = this._animators;
                    return t.push(e), e.done(function() {
                        var r = t.indexOf(e);
                        r >= 0 && t.splice(r, 1)
                    }), e
                }
            });
        Object.defineProperty(l.prototype, "autoRotate", { get: function(e) { return this._autoRotate }, set: function(e) { this._autoRotate = e, this._rotating = e } }), e.exports = l
    }, function(e, t) { e.exports = "@export ecgl.lines3D.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position: POSITION;\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n\nvoid main()\n{\n gl_Position = worldViewProjection * vec4(position, 1.0);\n v_Color = a_Color;\n}\n\n@end\n\n@export ecgl.lines3D.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nvarying vec4 v_Color;\n\n@import qtek.util.srgb\n\nvoid main()\n{\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(color * v_Color);\n#else\n gl_FragColor = color * v_Color;\n#endif\n}\n@end\n\n\n\n@export ecgl.lines3D.clipNear\n\nvec4 clipNear(vec4 p1, vec4 p2) {\n float n = (p1.w - near) / (p1.w - p2.w);\n return vec4(mix(p1.xy, p2.xy, n), -near, near);\n}\n\n@end\n\n@export ecgl.lines3D.expandLine\n#ifdef VERTEX_ANIMATION\n vec4 prevProj = worldViewProjection * vec4(mix(prevPositionPrev, positionPrev, percent), 1.0);\n vec4 currProj = worldViewProjection * vec4(mix(prevPosition, position, percent), 1.0);\n vec4 nextProj = worldViewProjection * vec4(mix(prevPositionNext, positionNext, percent), 1.0);\n#else\n vec4 prevProj = worldViewProjection * vec4(positionPrev, 1.0);\n vec4 currProj = worldViewProjection * vec4(position, 1.0);\n vec4 nextProj = worldViewProjection * vec4(positionNext, 1.0);\n#endif\n\n if (currProj.w < 0.0) {\n if (nextProj.w > 0.0) {\n currProj = clipNear(currProj, nextProj);\n }\n else if (prevProj.w > 0.0) {\n currProj = clipNear(currProj, prevProj);\n }\n }\n\n vec2 prevScreen = (prevProj.xy / abs(prevProj.w) + 1.0) * 0.5 * viewport.zw;\n vec2 currScreen = (currProj.xy / abs(currProj.w) + 1.0) * 0.5 * viewport.zw;\n vec2 nextScreen = (nextProj.xy / abs(nextProj.w) + 1.0) * 0.5 * viewport.zw;\n\n vec2 dir;\n float len = offset;\n if (position == positionPrev) {\n dir = normalize(nextScreen - currScreen);\n }\n else if (position == positionNext) {\n dir = normalize(currScreen - prevScreen);\n }\n else {\n vec2 dirA = normalize(currScreen - prevScreen);\n vec2 dirB = normalize(nextScreen - currScreen);\n\n vec2 tanget = normalize(dirA + dirB);\n\n float miter = 1.0 / max(dot(tanget, dirA), 0.5);\n len *= miter;\n dir = tanget;\n }\n\n dir = vec2(-dir.y, dir.x) * len;\n currScreen += dir;\n\n currProj.xy = (currScreen / viewport.zw - 0.5) * 2.0 * abs(currProj.w);\n@end\n\n\n@export ecgl.meshLines3D.vertex\n\nattribute vec3 position: POSITION;\nattribute vec3 positionPrev;\nattribute vec3 positionNext;\nattribute float offset;\nattribute vec4 a_Color : COLOR;\n\n#ifdef VERTEX_ANIMATION\nattribute vec3 prevPosition;\nattribute vec3 prevPositionPrev;\nattribute vec3 prevPositionNext;\nuniform float percent : 1.0;\n#endif\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform vec4 viewport : VIEWPORT;\nuniform float near : NEAR;\n\nvarying vec4 v_Color;\n\n@import ecgl.common.wireframe.vertexHeader\n\n@import ecgl.lines3D.clipNear\n\nvoid main()\n{\n @import ecgl.lines3D.expandLine\n\n gl_Position = currProj;\n\n v_Color = a_Color;\n\n @import ecgl.common.wireframe.vertexMain\n}\n@end\n\n\n@export ecgl.meshLines3D.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nvarying vec4 v_Color;\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.util.srgb\n\nvoid main()\n{\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(color * v_Color);\n#else\n gl_FragColor = color * v_Color;\n#endif\n\n @import ecgl.common.wireframe.fragmentMain\n}\n\n@end" }, function(e, t, r) {
        "use strict";

        function n(e, t, r, n, i) {
            var a = 0,
                o = 0;
            null == n && (n = 1 / 0), null == i && (i = 1 / 0);
            var s = 0;
            t.eachChild(function(u, h) {
                var l, c, d = u.position,
                    f = u.getBoundingRect(),
                    p = t.childAt(h + 1),
                    _ = p && p.getBoundingRect();
                if ("horizontal" === e) {
                    var m = f.width + (_ ? -_.x + f.x : 0);
                    l = a + m, l > n || u.newline ? (a = 0, l = m, o += s + r, s = f.height) : s = Math.max(s, f.height)
                } else {
                    var g = f.height + (_ ? -_.y + f.y : 0);
                    c = o + g, c > i || u.newline ? (a += s + r, o = 0, c = g, s = f.width) : s = Math.max(s, f.width)
                }
                u.newline || (d[0] = a, d[1] = o, "horizontal" === e ? a = l + r : o = c + r)
            })
        }
        var i = r(15),
            a = r(84),
            o = r(68),
            s = r(195),
            u = o.parsePercent,
            h = i.each,
            l = {},
            c = l.LOCATION_PARAMS = ["left", "right", "top", "bottom", "width", "height"],
            d = l.HV_NAMES = [
                ["width", "left", "right"],
                ["height", "top", "bottom"]
            ];
        l.box = n, l.vbox = i.curry(n, "vertical"), l.hbox = i.curry(n, "horizontal"), l.getAvailableSize = function(e, t, r) {
            var n = t.width,
                i = t.height,
                a = u(e.x, n),
                o = u(e.y, i),
                h = u(e.x2, n),
                l = u(e.y2, i);
            return (isNaN(a) || isNaN(parseFloat(e.x))) && (a = 0), (isNaN(h) || isNaN(parseFloat(e.x2))) && (h = n), (isNaN(o) || isNaN(parseFloat(e.y))) && (o = 0), (isNaN(l) || isNaN(parseFloat(e.y2))) && (l = i), r = s.normalizeCssArray(r || 0), { width: Math.max(h - a - r[1] - r[3], 0), height: Math.max(l - o - r[0] - r[2], 0) }
        }, l.getLayoutRect = function(e, t, r) {
            r = s.normalizeCssArray(r || 0);
            var n = t.width,
                i = t.height,
                o = u(e.left, n),
                h = u(e.top, i),
                l = u(e.right, n),
                c = u(e.bottom, i),
                d = u(e.width, n),
                f = u(e.height, i),
                p = r[2] + r[0],
                _ = r[1] + r[3],
                m = e.aspect;
            switch (isNaN(d) && (d = n - l - _ - o), isNaN(f) && (f = i - c - p - h), null != m && (isNaN(d) && isNaN(f) && (m > n / i ? d = .8 * n : f = .8 * i), isNaN(d) && (d = m * f), isNaN(f) && (f = d / m)), isNaN(o) && (o = n - l - d - _), isNaN(h) && (h = i - c - f - p), e.left || e.right) {
                case "center":
                    o = n / 2 - d / 2 - r[3];
                    break;
                case "right":
                    o = n - d - _
            }
            switch (e.top || e.bottom) {
                case "middle":
                case "center":
                    h = i / 2 - f / 2 - r[0];
                    break;
                case "bottom":
                    h = i - f - p
            }
            o = o || 0, h = h || 0, isNaN(d) && (d = n - _ - o - (l || 0)), isNaN(f) && (f = i - p - h - (c || 0));
            var g = new a(o + r[3], h + r[0], d, f);
            return g.margin = r, g
        }, l.positionElement = function(e, t, r, n, o) {
            var s = !o || !o.hv || o.hv[0],
                u = !o || !o.hv || o.hv[1],
                h = o && o.boundingMode || "all";
            if (s || u) {
                var c;
                if ("raw" === h) c = "group" === e.type ? new a(0, 0, +t.width || 0, +t.height || 0) : e.getBoundingRect();
                else if (c = e.getBoundingRect(), e.needLocalTransform()) {
                    var d = e.getLocalTransform();
                    c = c.clone(), c.applyTransform(d)
                }
                t = l.getLayoutRect(i.defaults({ width: c.width, height: c.height }, t), r, n);
                var f = e.position,
                    p = s ? t.x - c.x : 0,
                    _ = u ? t.y - c.y : 0;
                e.attr("position", "raw" === h ? [p, _] : [f[0] + p, f[1] + _])
            }
        }, l.sizeCalculable = function(e, t) { return null != e[d[t][0]] || null != e[d[t][1]] && null != e[d[t][2]] }, l.mergeLayoutParam = function(e, t, r) {
            function n(r, n) {
                var i = {},
                    s = 0,
                    l = {},
                    c = 0;
                if (h(r, function(t) { l[t] = e[t] }), h(r, function(e) { a(t, e) && (i[e] = l[e] = t[e]), o(i, e) && s++, o(l, e) && c++ }), u[n]) return o(t, r[1]) ? l[r[2]] = null : o(t, r[2]) && (l[r[1]] = null), l;
                if (2 !== c && s) { if (s >= 2) return i; for (var d = 0; d < r.length; d++) { var f = r[d]; if (!a(i, f) && a(e, f)) { i[f] = e[f]; break } } return i }
                return l
            }

            function a(e, t) { return e.hasOwnProperty(t) }

            function o(e, t) { return null != e[t] && "auto" !== e[t] }

            function s(e, t, r) { h(e, function(e) { t[e] = r[e] }) }!i.isObject(r) && (r = {});
            var u = r.ignoreSize;
            !i.isArray(u) && (u = [u, u]);
            var l = n(d[0], 0),
                c = n(d[1], 1);
            s(d[0], e, l), s(d[1], e, c)
        }, l.getLayoutParams = function(e) { return l.copyLayoutParams({}, e) }, l.copyLayoutParams = function(e, t) { return t && e && h(c, function(r) { t.hasOwnProperty(r) && (e[r] = t[r]) }), e }, e.exports = l
    }, function(e, t) {
        e.exports = function(e, t, r, n, i) {
            n.eachRawSeriesByType(e, function(e) {
                var i = e.getData(),
                    a = e.get("symbol") || t,
                    o = e.get("symbolSize");
                i.setVisual({ legendSymbol: r || a, symbol: a, symbolSize: o }), n.isSeriesFiltered(e) || ("function" == typeof o && i.each(function(t) {
                    var r = e.getRawValue(t),
                        n = e.getDataParams(t);
                    i.setItemVisual(t, "symbolSize", o(r, n))
                }), i.each(function(e) {
                    var t = i.getItemModel(e),
                        r = t.getShallow("symbol", !0),
                        n = t.getShallow("symbolSize", !0);
                    null != r && i.setItemVisual(e, "symbol", r), null != n && i.setItemVisual(e, "symbolSize", n)
                }))
            })
        }
    }, function(e, t, r) {
        "use strict";
        var n = r(69),
            i = n.extend({ fov: 50, aspect: 1, near: .1, far: 2e3 }, {
                updateProjectionMatrix: function() {
                    var e = this.fov / 180 * Math.PI;
                    this.projectionMatrix.perspective(e, this.aspect, this.near, this.far)
                },
                decomposeProjectionMatrix: function() {
                    var e = this.projectionMatrix._array,
                        t = 2 * Math.atan(1 / e[5]);
                    this.fov = t / Math.PI * 180, this.aspect = e[5] / e[0], this.near = e[14] / (e[10] - 1), this.far = e[14] / (e[10] + 1)
                },
                clone: function() { var e = n.prototype.clone.call(this); return e.fov = this.fov, e.aspect = this.aspect, e.near = this.near, e.far = this.far, e }
            });
        e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = function() { this._contextId = 0, this._caches = [], this._context = {} };
        n.prototype = {
            use: function(e, t) {
                var r = this._caches;
                r[e] || (r[e] = {}, t && (r[e] = t())), this._contextId = e, this._context = r[e]
            },
            put: function(e, t) { this._context[e] = t },
            get: function(e) { return this._context[e] },
            dirty: function(e) {
                e = e || "";
                var t = "__dt__" + e;
                this.put(t, !0)
            },
            dirtyAll: function(e) { e = e || ""; for (var t = "__dt__" + e, r = this._caches, n = 0; n < r.length; n++) r[n] && (r[n][t] = !0) },
            fresh: function(e) {
                e = e || "";
                var t = "__dt__" + e;
                this.put(t, !1)
            },
            freshAll: function(e) { e = e || ""; for (var t = "__dt__" + e, r = this._caches, n = 0; n < r.length; n++) r[n] && (r[n][t] = !1) },
            isDirty: function(e) {
                e = e || "";
                var t = "__dt__" + e,
                    r = this._context;
                return !r.hasOwnProperty(t) || !0 === r[t]
            },
            deleteContext: function(e) { delete this._caches[e], this._context = {} },
            delete: function(e) { delete this._context[e] },
            clearAll: function() { this._caches = {} },
            getContext: function() { return this._context },
            eachContext: function(e, t) { Object.keys(this._caches).forEach(function(r) { e && e.call(t, r) }) },
            miss: function(e) { return !this._context.hasOwnProperty(e) }
        }, n.prototype.constructor = n, e.exports = n
    }, function(e, t, r) {
        "use strict";
        var n = r(13),
            i = r(14),
            a = n.extend({ widthSegments: 1, heightSegments: 1 }, function() { this.build() }, {
                build: function() {
                    for (var e = this.heightSegments, t = this.widthSegments, r = this.attributes, n = [], a = [], o = [], s = [], u = 0; u <= e; u++)
                        for (var h = u / e, l = 0; l <= t; l++) {
                            var c = l / t;
                            if (n.push([2 * c - 1, 2 * h - 1, 0]), a && a.push([c, h]), o && o.push([0, 0, 1]), l < t && u < e) {
                                var d = l + u * (t + 1);
                                s.push([d, d + 1, d + t + 1]), s.push([d + t + 1, d + 1, d + t + 2])
                            }
                        }
                    r.position.fromArray(n), r.texcoord0.fromArray(a), r.normal.fromArray(o), this.initIndicesFromArray(s), this.boundingBox = new i, this.boundingBox.min.set(-1, -1, 0), this.boundingBox.max.set(1, 1, 0)
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(5),
            i = r(23),
            a = r(73),
            o = r(59),
            s = r(58),
            u = r(26),
            h = r(229),
            l = r(230),
            c = {
                loadTexture: function(e, t, r, a) {
                    var o;
                    if ("function" == typeof t ? (r = t, a = r, t = {}) : t = t || {}, "string" == typeof e) {
                        if (e.match(/.hdr$/) || "hdr" === t.fileType) return o = new n({ width: 0, height: 0 }), c._fetchTexture(e, function(e) { l.parseRGBE(e, o, t.exposure), o.dirty(), r && r(o) }, a), o;
                        e.match(/.dds$/) || "dds" === t.fileType ? (o = new n({ width: 0, height: 0 }), c._fetchTexture(e, function(e) { h.parse(e, o), o.dirty(), r && r(o) }, a)) : (o = new n, o.load(e), o.success(r), o.error(a))
                    } else if ("object" == typeof e && void 0 !== e.px) {
                        var o = new i;
                        o.load(e), o.success(r), o.error(a)
                    }
                    return o
                },
                loadPanorama: function(e, t, r, n, i, a) { var o = this; "function" == typeof n ? (i = n, a = i, n = {}) : n = n || {}, c.loadTexture(t, n, function(t) { t.flipY = n.flipY || !1, o.panoramaToCubeMap(e, t, r, n), t.dispose(e.gl), i && i(r) }, a) },
                panoramaToCubeMap: function(e, t, r, n) {
                    var i = new o,
                        a = new s({ scene: new u });
                    return a.material.set("diffuseMap", t), n = n || {}, n.encodeRGBM && a.material.shader.define("fragment", "RGBM_ENCODE"), i.texture = r, i.render(e, a.scene), i.texture = null, i.dispose(e), r
                },
                _fetchTexture: function(e, t, r) { a.get({ url: e, responseType: "arraybuffer", onload: t, onerror: r }) },
                createChessboard: function(e, t, r, i) {
                    e = e || 512, t = t || 64, r = r || "black", i = i || "white";
                    var a = Math.ceil(e / t),
                        o = document.createElement("canvas");
                    o.width = e, o.height = e;
                    var s = o.getContext("2d");
                    s.fillStyle = i, s.fillRect(0, 0, e, e), s.fillStyle = r;
                    for (var u = 0; u < a; u++)
                        for (var h = 0; h < a; h++) {
                            var l = h % 2 ? u % 2 : u % 2 - 1;
                            l && s.fillRect(u * t, h * t, t, t)
                        }
                    return new n({ image: o, anisotropic: 8 })
                },
                createBlank: function(e) {
                    var t = document.createElement("canvas");
                    t.width = 1, t.height = 1;
                    var r = t.getContext("2d");
                    return r.fillStyle = e, r.fillRect(0, 0, 1, 1), new n({ image: t })
                }
            };
        e.exports = c
    }, function(e, t, r) {
        function n(e, t) { return e && t && e[0] === t[0] && e[1] === t[1] }

        function i(e, t) { this.rootNode = new o.Node, this.is2D = e, this._labelsBuilder = new h(256, 256, t), this._labelsBuilder.getMesh().renderOrder = 100, this.rootNode.add(this._labelsBuilder.getMesh()), this._api = t, this._spriteImageCanvas = document.createElement("canvas") }
        var a = r(0),
            o = r(2),
            s = r(188),
            u = r(105),
            h = r(49),
            l = r(9);
        i.prototype = {
            constructor: i,
            highlightOnMouseover: !0,
            update: function(e, t, r) {
                var i = this._prevMesh;
                if (this._prevMesh = this._mesh, this._mesh = i, !this._mesh) {
                    var a = this._prevMesh && this._prevMesh.material;
                    this._mesh = new u({ renderOrder: 10 }), a && (this._mesh.material = a)
                }
                this.rootNode.remove(this._prevMesh), this.rootNode.add(this._mesh), this._setPositionTextureToMesh(this._mesh, this._positionTexture);
                var h = e.getData(),
                    l = this._getSymbolInfo(h),
                    c = r.getDevicePixelRatio();
                l.maxSize = Math.min(2 * l.maxSize, 200);
                var d = [];
                l.aspect > 1 ? (d[0] = l.maxSize, d[1] = l.maxSize / l.aspect) : (d[1] = l.maxSize, d[0] = l.maxSize * l.aspect);
                var f = e.getModel("itemStyle").getItemStyle();
                d[0] = d[0] || 1, d[1] = d[1] || 1, this._symbolType === l.type && n(this._symbolSize, d) && this._lineWidth === f.lineWidth || (s.createSymbolSprite(l.type, d, { fill: "#fff", lineWidth: f.lineWidth, stroke: "transparent", shadowColor: "transparent", minMargin: Math.min(d[0] / 2, 10) }, this._spriteImageCanvas), s.createSDFFromCanvas(this._spriteImageCanvas, Math.min(this._spriteImageCanvas.width, 32), 20, this._mesh.material.get("sprite").image), this._symbolType = l.type, this._symbolSize = d, this._lineWidth = f.lineWidth);
                var p = this._mesh.geometry,
                    _ = h.getLayout("points"),
                    m = p.attributes;
                m.position.init(h.count()), m.size.init(h.count()), m.color.init(h.count());
                var g = m.position.value,
                    v = [],
                    y = this.is2D,
                    x = this._spriteImageCanvas.width / l.maxSize * c;
                this._originalOpacity = new Float32Array(h.count());
                for (var T = 0; T < h.count(); T++) {
                    var b = 3 * T,
                        w = 2 * T;
                    y ? (g[b] = _[w], g[b + 1] = _[w + 1], g[b + 2] = -10) : (g[b] = _[b], g[b + 1] = _[b + 1], g[b + 2] = _[b + 2]);
                    var E = h.getItemVisual(T, "color"),
                        S = h.getItemVisual(T, "opacity");
                    o.parseColor(E, v), v[3] *= S, this._originalOpacity[T] = v[3], m.color.set(T, v), v[3] < .99 && !0;
                    var d = h.getItemVisual(T, "symbolSize");
                    d = d instanceof Array ? Math.max(d[0], d[1]) : d, isNaN(d) && (d = 0), m.size.value[T] = d * x
                }
                this._mesh.sizeScale = x, p.dirty();
                var A = "lighter" === e.get("blendMode") ? o.additiveBlend : null,
                    a = this._mesh.material;
                a.blend = A, a.set("lineWidth", f.lineWidth / 20);
                var M = o.parseColor(f.stroke);
                a.set("color", [1, 1, 1, 1]), a.set("strokeColor", M), this.is2D ? (a.transparent = !0, a.depthMask = !1, a.depthTest = !1, p.sortVertices = !1) : (a.depthTest = !0, a.transparent = !0, a.depthMask = !1, p.sortVertices = !0);
                var N = e.coordinateSystem;
                if (N && N.viewGL) {
                    var C = N.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._mesh.material.shader[C]("fragment", "SRGB_DECODE")
                }
                this._updateHandler(e, t, r), this._labelsBuilder.updateData(h), this._labelsBuilder.getLabelPosition = function(e, t, r) { var n = 3 * e; return [g[n], g[n + 1], g[n + 2]] }, this._labelsBuilder.getLabelDistance = function(e, t, r) { return p.attributes.size.get(e) / x / 2 + r }, this._labelsBuilder.updateLabels(), this._updateAnimation(e), this._api = r
            },
            _updateAnimation: function(e) {
                o.updateVertexAnimation([
                    ["prevPosition", "position"],
                    ["prevSize", "size"]
                ], this._prevMesh, this._mesh, e)
            },
            _updateHandler: function(e, t, r) {
                var n, i = e.getData(),
                    a = this._mesh,
                    o = -1,
                    s = e.coordinateSystem && "cartesian3D" === e.coordinateSystem.type;
                s && (n = e.coordinateSystem.model), a.seriesIndex = e.seriesIndex, a.off("mousemove"), a.off("mouseout"), a.on("mousemove", function(e) {
                    var t = e.vertexIndex;
                    t !== o && (this.highlightOnMouseover && (this.downplay(i, o), this.highlight(i, t), this._labelsBuilder.updateLabels([t])), s && r.dispatchAction({ type: "grid3DShowAxisPointer", value: [i.get("x", t), i.get("y", t), i.get("z", t)], grid3DIndex: n.componentIndex })), a.dataIndex = t, o = t
                }, this), a.on("mouseout", function(e) { this.highlightOnMouseover && (this.downplay(i, e.vertexIndex), this._labelsBuilder.updateLabels()), o = -1, a.dataIndex = -1, s && r.dispatchAction({ type: "grid3DHideAxisPointer", grid3DIndex: n.componentIndex }) }, this)
            },
            updateView: function(e) {
                if (this._mesh) {
                    var t = new l;
                    l.mul(t, e.viewMatrix, this._mesh.worldTransform), l.mul(t, e.projectionMatrix, t), this._mesh.updateNDCPosition(t, this.is2D, this._api)
                }
            },
            updateLayout: function(e, t, r) {
                var n = e.getData();
                if (this._mesh) {
                    var i = this._mesh.geometry.attributes.position.value,
                        a = n.getLayout("points");
                    if (this.is2D)
                        for (var o = 0; o < a.length / 2; o++) {
                            var s = 3 * o,
                                u = 2 * o;
                            i[s] = a[u], i[s + 1] = a[u + 1], i[s + 2] = -10
                        } else
                            for (var o = 0; o < a.length; o++) i[o] = a[o];
                    this._mesh.geometry.dirty(), r.getZr().refresh()
                }
            },
            highlight: function(e, t) {
                var r = e.getItemModel(t),
                    n = r.getModel("emphasis.itemStyle"),
                    i = n.get("color"),
                    s = n.get("opacity");
                if (null == i) {
                    var u = e.getItemVisual(t, "color");
                    i = a.color.lift(u, -.4)
                }
                null == s && (s = e.getItemVisual(t, "opacity"));
                var h = o.parseColor(i);
                h[3] *= s, this._mesh.geometry.attributes.color.set(t, h), this._mesh.geometry.dirtyAttribute("color"), this._api.getZr().refresh()
            },
            downplay: function(e, t) {
                var r = e.getItemVisual(t, "color"),
                    n = e.getItemVisual(t, "opacity"),
                    i = o.parseColor(r);
                i[3] *= n, this._mesh.geometry.attributes.color.set(t, i), this._mesh.geometry.dirtyAttribute("color"), this._api.getZr().refresh()
            },
            fadeOutAll: function(e) {
                for (var t = this._mesh.geometry, r = 0; r < t.vertexCount; r++) {
                    var n = this._originalOpacity[r] * e;
                    t.attributes.color.value[4 * r + 3] = n
                }
                t.dirtyAttribute("color"), this._api.getZr().refresh()
            },
            fadeInAll: function() { this.fadeOutAll(1) },
            setPositionTexture: function(e) { this._mesh && this._setPositionTextureToMesh(this._mesh, e), this._positionTexture = e },
            removePositionTexture: function() { this._positionTexture = null, this._mesh && this._setPositionTextureToMesh(this._mesh, null) },
            _setPositionTextureToMesh: function(e, t) { t && e.material.set("positionTexture", t), e.material.shader[t ? "enableTexture" : "disableTexture"]("positionTexture") },
            getPointsMesh: function() { return this._mesh },
            updateLabels: function(e) { this._labelsBuilder.updateLabels(e) },
            hideLabels: function() { this.rootNode.remove(this._labelsBuilder.getMesh()) },
            showLabels: function() { this.rootNode.add(this._labelsBuilder.getMesh()) },
            _getSymbolInfo: function(e) {
                var t, r = e.getItemVisual(0, "symbol") || "circle",
                    n = 0;
                return e.each(function(i) {
                    var a, o = e.getItemVisual(i, "symbolSize"),
                        s = e.getItemVisual(i, "symbol");
                    if (o instanceof Array) a = o[0] / o[1], n = Math.max(Math.max(o[0], o[1]), n);
                    else {
                        if (isNaN(o)) return;
                        a = 1, n = Math.max(o, n)
                    }
                    r = s, t = a
                }), { maxSize: n, type: r, aspect: t }
            }
        }, e.exports = i
    }, function(e, t, r) {
        function n(e, t, r) { this._labelsMesh = new o, this._labelTextureSurface = new a({ width: 512, height: 512, devicePixelRatio: r.getDevicePixelRatio(), onupdate: function() { r.getZr().refresh() } }), this._api = r, this._labelsMesh.material.set("textureAtlas", this._labelTextureSurface.getTexture()) }
        var i = r(0),
            a = r(67),
            o = r(51),
            s = r(4);
        n.prototype.getLabelPosition = function(e, t, r) { return [0, 0, 0] }, n.prototype.getLabelDistance = function(e, t, r) { return 0 }, n.prototype.getMesh = function() { return this._labelsMesh }, n.prototype.updateData = function(e) {
            this._labelsVisibilitiesBits && this._labelsVisibilitiesBits.length === e.count() || (this._labelsVisibilitiesBits = new Uint8Array(e.count()));
            var t = ["label", "show"],
                r = ["emphasis", "label", "show"];
            e.each(function(n) {
                var i = e.getItemModel(n),
                    a = i.get(t),
                    o = i.get(r);
                null == o && (o = a);
                var s = (a ? 1 : 0) | (o ? 2 : 0);
                this._labelsVisibilitiesBits[n] = s
            }, !1, this), this._data = e
        }, n.prototype.updateLabels = function(e) {
            if (this._data) {
                e = e || [];
                for (var t = e.length > 0, r = {}, n = 0; n < e.length; n++) r[e[n]] = !0;
                this._labelsMesh.geometry.convertToDynamicArray(!0), this._labelTextureSurface.clear();
                var a = ["label"],
                    o = ["emphasis", "label"],
                    u = this._data.hostModel,
                    h = this._data,
                    l = u.getModel(a),
                    c = u.getModel(o, l),
                    d = { left: "right", right: "left", top: "center", bottom: "center" },
                    f = { left: "middle", right: "middle", top: "bottom", bottom: "top" };
                h.each(function(e) {
                    var n = !1;
                    if (t && r[e] && (n = !0), this._labelsVisibilitiesBits[e] & (n ? 2 : 1)) {
                        var p = h.getItemModel(e),
                            _ = p.getModel(n ? o : a, n ? c : l),
                            m = _.get("distance") || 0,
                            g = _.get("position"),
                            v = _.getModel("textStyle"),
                            y = this._api.getDevicePixelRatio(),
                            x = u.getFormattedLabel(e, n ? "emphasis" : "normal");
                        if (null != x && "" !== x) {
                            var T = new i.graphic.Text;
                            i.graphic.setTextStyle(T.style, v, { text: x, textFill: v.get("color") || h.getItemVisual(e, "color") || "#000", textAlign: "left", textVerticalAlign: "top", opacity: s.firstNotNull(v.get("opacity"), h.getItemVisual(e, "opacity"), 1) });
                            var b = T.getBoundingRect();
                            b.height *= 1.2;
                            var w = this._labelTextureSurface.add(T),
                                E = d[g] || "center",
                                S = f[g] || "bottom";
                            this._labelsMesh.geometry.addSprite(this.getLabelPosition(e, g, m), [b.width * y, b.height * y], w, E, S, this.getLabelDistance(e, g, m) * y)
                        }
                    }
                }, !1, this), this._labelsMesh.material.set("uvScale", this._labelTextureSurface.getCoordsScale()), this._labelTextureSurface.getZr().refreshImmediately(), this._labelsMesh.geometry.convertToTypedArray(), this._labelsMesh.geometry.dirty()
            }
        }, e.exports = n
    }, function(e, t, r) {
        var n = r(1).vec3,
            i = r(66),
            a = n.create(),
            o = n.create(),
            s = n.create();
        e.exports = {
            needsSortTriangles: function() { return this.indices && this.sortTriangles },
            needsSortTrianglesProgressively: function() { return this.needsSortTriangles() && this.triangleCount >= 2e4 },
            doSortTriangles: function(e, t) {
                var r = this.indices;
                if (0 === t) {
                    var i = this.attributes.position,
                        e = e._array;
                    this._triangleZList && this._triangleZList.length === this.triangleCount || (this._triangleZList = new Float32Array(this.triangleCount), this._sortedTriangleIndices = new Uint32Array(this.triangleCount), this._indicesTmp = new r.constructor(r.length), this._triangleZListTmp = new Float32Array(this.triangleCount));
                    for (var u, h = 0, l = 0; l < r.length;) {
                        i.get(r[l++], a), i.get(r[l++], o), i.get(r[l++], s);
                        var c = n.sqrDist(a, e),
                            d = n.sqrDist(o, e),
                            f = n.sqrDist(s, e),
                            p = Math.min(c, d);
                        p = Math.min(p, f), 3 === l ? (u = p, p = 0) : p -= u, this._triangleZList[h++] = p
                    }
                }
                for (var _ = this._sortedTriangleIndices, l = 0; l < _.length; l++) _[l] = l;
                if (this.triangleCount < 2e4) 0 === t && this._simpleSort(!0);
                else
                    for (var l = 0; l < 3; l++) this._progressiveQuickSort(3 * t + l);
                for (var m = this._indicesTmp, g = this._triangleZListTmp, v = this._triangleZList, l = 0; l < this.triangleCount; l++) {
                    var y = 3 * _[l],
                        x = 3 * l;
                    m[x++] = r[y++], m[x++] = r[y++], m[x] = r[y], g[l] = v[_[l]]
                }
                var T = this._indicesTmp;
                this._indicesTmp = this.indices, this.indices = T;
                var T = this._triangleZListTmp;
                this._triangleZListTmp = this._triangleZList, this._triangleZList = T, this.dirtyIndices()
            },
            _simpleSort: function(e) {
                function t(e, t) { return r[t] - r[e] }
                var r = this._triangleZList,
                    n = this._sortedTriangleIndices;
                e ? Array.prototype.sort.call(n, t) : i.sort(n, t, 0, n.length - 1)
            },
            _progressiveQuickSort: function(e) {
                var t = this._triangleZList,
                    r = this._sortedTriangleIndices;
                this._quickSort = this._quickSort || new i, this._quickSort.step(r, function(e, r) { return t[r] - t[e] }, e)
            }
        }
    }, function(e, t, r) {
        var n = r(2),
            i = r(176);
        n.Shader.import(r(182)), e.exports = n.Mesh.extend(function() { return { geometry: new i({ dynamic: !0 }), material: new n.Material({ shader: n.createShader("ecgl.labels"), transparent: !0, depthMask: !1 }), culling: !1, castShadow: !1, ignorePicking: !0 } })
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(17),
            a = r(11),
            o = (r(20), r(14)),
            s = r(9),
            u = (r(81), r(16)),
            h = r(28),
            l = r(7);
        l.import(r(224)), l.import(r(82));
        var c = r(1),
            d = c.mat4,
            f = c.vec3,
            p = d.create,
            _ = 0,
            m = {},
            g = n.extend(function() { return { canvas: null, _width: 100, _height: 100, devicePixelRatio: window.devicePixelRatio || 1, clearColor: [0, 0, 0, 0], clearBit: 17664, alpha: !0, depth: !0, stencil: !1, antialias: !0, premultipliedAlpha: !0, preserveDrawingBuffer: !1, throwError: !0, gl: null, viewport: {}, __currentFrameBuffer: null, _viewportStack: [], _clearStack: [], _sceneRendering: null } }, function() {
                this.canvas || (this.canvas = document.createElement("canvas"));
                var e = this.canvas;
                try {
                    var t = { alpha: this.alpha, depth: this.depth, stencil: this.stencil, antialias: this.antialias, premultipliedAlpha: this.premultipliedAlpha, preserveDrawingBuffer: this.preserveDrawingBuffer };
                    if (this.gl = e.getContext("webgl", t) || e.getContext("experimental-webgl", t), !this.gl) throw new Error;
                    null == this.gl.__GLID__ && (this.gl.__GLID__ = _++, i.initialize(this.gl)), this.resize()
                } catch (e) { throw "Error creating WebGL Context " + e }
            }, {
                resize: function(e, t) {
                    var r = this.canvas,
                        n = this.devicePixelRatio;
                    null != e ? (r.style.width = e + "px", r.style.height = t + "px", r.width = e * n, r.height = t * n, this._width = e, this._height = t) : (this._width = r.width / n, this._height = r.height / n), this.setViewport(0, 0, this._width, this._height)
                },
                getWidth: function() { return this._width },
                getHeight: function() { return this._height },
                getViewportAspect: function() { var e = this.viewport; return e.width / e.height },
                setDevicePixelRatio: function(e) { this.devicePixelRatio = e, this.resize(this._width, this._height) },
                getDevicePixelRatio: function() { return this.devicePixelRatio },
                getExtension: function(e) { return i.getExtension(this.gl, e) },
                setViewport: function(e, t, r, n, i) {
                    if ("object" == typeof e) {
                        var a = e;
                        e = a.x, t = a.y, r = a.width, n = a.height, i = a.devicePixelRatio
                    }
                    i = i || this.devicePixelRatio, this.gl.viewport(e * i, t * i, r * i, n * i), this.viewport = { x: e, y: t, width: r, height: n, devicePixelRatio: i }
                },
                saveViewport: function() { this._viewportStack.push(this.viewport) },
                restoreViewport: function() { this._viewportStack.length > 0 && this.setViewport(this._viewportStack.pop()) },
                saveClear: function() { this._clearStack.push({ clearBit: this.clearBit, clearColor: this.clearColor }) },
                restoreClear: function() {
                    if (this._clearStack.length > 0) {
                        var e = this._clearStack.pop();
                        this.clearColor = e.clearColor, this.clearBit = e.clearBit
                    }
                },
                bindSceneRendering: function(e) { this._sceneRendering = e },
                beforeRenderObject: function() {},
                afterRenderObject: function() {},
                render: function(e, t, r, n) {
                    var i = this.gl;
                    this._sceneRendering = e;
                    var a = this.clearColor;
                    if (this.clearBit) {
                        i.colorMask(!0, !0, !0, !0), i.depthMask(!0);
                        var o = this.viewport,
                            s = !1,
                            u = o.devicePixelRatio;
                        (o.width !== this._width || o.height !== this._height || u && u !== this.devicePixelRatio || o.x || o.y) && (s = !0, i.enable(i.SCISSOR_TEST), i.scissor(o.x * u, o.y * u, o.width * u, o.height * u)), i.clearColor(a[0], a[1], a[2], a[3]), i.clear(this.clearBit), s && i.disable(i.SCISSOR_TEST)
                    }
                    r || e.update(!1), t.getScene() || t.update(!0);
                    for (var h = e.opaqueQueue, l = e.transparentQueue, c = e.material, _ = 0; _ < h.length; _++) {
                        var m = h[_].material;
                        m.updateShader && m.updateShader(i)
                    }
                    for (var _ = 0; _ < l.length; _++) {
                        var m = l[_].material;
                        m.updateShader && m.updateShader(i)
                    }
                    if (e.trigger("beforerender", this, e, t), l.length > 0)
                        for (var g = p(), v = f.create(), _ = 0; _ < l.length; _++) {
                            var y = l[_];
                            d.multiplyAffine(g, t.viewMatrix._array, y.worldTransform._array), f.transformMat4(v, y.position._array, g), y.__depth = v[2]
                        }
                    h.sort(this.opaqueSortFunc), l.sort(this.transparentSortFunc), e.trigger("beforerender:opaque", this, h), e.viewBoundingBoxLastFrame.min.set(1 / 0, 1 / 0, 1 / 0), e.viewBoundingBoxLastFrame.max.set(-1 / 0, -1 / 0, -1 / 0), i.disable(i.BLEND), i.enable(i.DEPTH_TEST);
                    var x = this.renderQueue(h, t, c, n);
                    e.trigger("afterrender:opaque", this, h, x), e.trigger("beforerender:transparent", this, l), i.enable(i.BLEND);
                    var T = this.renderQueue(l, t, c);
                    e.trigger("afterrender:transparent", this, l, T);
                    var b = {};
                    for (var w in x) b[w] = x[w] + T[w];
                    return e.trigger("afterrender", this, e, t, b), this._sceneRendering = null, b
                },
                resetRenderStatus: function() { this._currentShader = null },
                ifRenderObject: function(e) { return !0 },
                renderQueue: function(e, t, r, n) {
                    var i = { triangleCount: 0, vertexCount: 0, drawCallCount: 0, meshCount: e.length, renderedMeshCount: 0 },
                        a = this.viewport,
                        o = a.devicePixelRatio,
                        s = [a.x * o, a.y * o, a.width * o, a.height * o],
                        u = this.devicePixelRatio,
                        h = this.__currentFrameBuffer ? [this.__currentFrameBuffer.getTextureWidth(), this.__currentFrameBuffer.getTextureHeight()] : [this._width * u, this._height * u],
                        l = [s[2], s[3]],
                        c = Date.now();
                    d.copy(v.VIEW, t.viewMatrix._array), d.copy(v.PROJECTION, t.projectionMatrix._array), d.multiply(v.VIEWPROJECTION, t.projectionMatrix._array, v.VIEW), d.copy(v.VIEWINVERSE, t.worldTransform._array), d.invert(v.PROJECTIONINVERSE, v.PROJECTION), d.invert(v.VIEWPROJECTIONINVERSE, v.VIEWPROJECTION);
                    var f, p, _, g = this.gl,
                        y = this._sceneRendering;
                    n ? _ = this._renderPreZ(e, y, t) : (_ = e, g.depthFunc(g.LESS));
                    for (var x, T, b, w, E, S = 0; S < _.length; S++) {
                        var A = _[S];
                        if (this.ifRenderObject(A)) {
                            var M = A.geometry,
                                N = A.worldTransform._array;
                            if (d.multiplyAffine(v.WORLDVIEW, v.VIEW, N), !M.boundingBox || n || !this.isFrustumCulled(A, y, t, v.WORLDVIEW, v.PROJECTION)) {
                                var C = r || A.material,
                                    L = C.shader;
                                d.copy(v.WORLD, N), d.multiply(v.WORLDVIEWPROJECTION, v.VIEWPROJECTION, N), (L.matrixSemantics.WORLDINVERSE || L.matrixSemantics.WORLDINVERSETRANSPOSE) && d.invert(v.WORLDINVERSE, N), (L.matrixSemantics.WORLDVIEWINVERSE || L.matrixSemantics.WORLDVIEWINVERSETRANSPOSE) && d.invert(v.WORLDVIEWINVERSE, v.WORLDVIEW), (L.matrixSemantics.WORLDVIEWPROJECTIONINVERSE || L.matrixSemantics.WORLDVIEWPROJECTIONINVERSETRANSPOSE) && d.invert(v.WORLDVIEWPROJECTIONINVERSE, v.WORLDVIEWPROJECTION), A.beforeRender(g), this.beforeRenderObject(A, f, p);
                                if (!L.isEqual(p)) {
                                    y && y.isShaderLightNumberChanged(L) && y.setShaderLightNumber(L);
                                    var D = L.bind(g);
                                    if (D) {
                                        if (m[L.__GUID__]) continue;
                                        if (m[L.__GUID__] = !0, this.throwError) throw new Error(D);
                                        this.trigger("error", D)
                                    }
                                    L.setUniformOfSemantic(g, "VIEWPORT", s), L.setUniformOfSemantic(g, "WINDOW_SIZE", h), L.setUniformOfSemantic(g, "NEAR", t.near), L.setUniformOfSemantic(g, "FAR", t.far), L.setUniformOfSemantic(g, "DEVICEPIXELRATIO", o), L.setUniformOfSemantic(g, "TIME", c), L.setUniformOfSemantic(g, "VIEWPORT_SIZE", l), y && y.setLightUniforms(L, g)
                                } else L = p;
                                f !== C && (n || (C.depthTest !== x && (C.depthTest ? g.enable(g.DEPTH_TEST) : g.disable(g.DEPTH_TEST), x = C.depthTest), C.depthMask !== T && (g.depthMask(C.depthMask), T = C.depthMask)), C.bind(g, L, f, p), f = C, C.transparent && (C.blend ? C.blend(g) : (g.blendEquationSeparate(g.FUNC_ADD, g.FUNC_ADD), g.blendFuncSeparate(g.SRC_ALPHA, g.ONE_MINUS_SRC_ALPHA, g.ONE, g.ONE_MINUS_SRC_ALPHA))));
                                for (var I = L.matrixSemanticKeys, R = 0; R < I.length; R++) {
                                    var P = I[R],
                                        O = L.matrixSemantics[P],
                                        F = v[P];
                                    if (O.isTranspose) {
                                        var B = v[O.semanticNoTranspose];
                                        d.transpose(F, B)
                                    }
                                    L.setUniform(g, O.type, O.symbol, F)
                                }
                                A.cullFace !== w && (w = A.cullFace, g.cullFace(w)), A.frontFace !== E && (E = A.frontFace, g.frontFace(E)), A.culling !== b && (b = A.culling, b ? g.enable(g.CULL_FACE) : g.disable(g.CULL_FACE));
                                var U = A.render(g, L);
                                U && (i.triangleCount += U.triangleCount, i.vertexCount += U.vertexCount, i.drawCallCount += U.drawCallCount, i.renderedMeshCount++), this.afterRenderObject(A, U), A.afterRender(g, U), p = L
                            }
                        }
                    }
                    return i
                },
                _renderPreZ: function(e, t, r) {
                    var n = this.gl,
                        i = this._prezMaterial || new u({ shader: new l({ vertex: l.source("qtek.prez.vertex"), fragment: l.source("qtek.prez.fragment") }) });
                    this._prezMaterial = i;
                    var a, o, s, h = i.shader,
                        c = [];
                    h.bind(n), n.colorMask(!1, !1, !1, !1), n.depthMask(!0), n.enable(n.DEPTH_TEST);
                    for (var f = 0; f < e.length; f++) {
                        var p = e[f];
                        if (this.ifRenderObject(p)) {
                            var _ = p.worldTransform._array,
                                m = p.geometry;
                            if (d.multiplyAffine(v.WORLDVIEW, v.VIEW, _), !(m.boundingBox && this.isFrustumCulled(p, t, r, v.WORLDVIEW, v.PROJECTION) || (c.push(p), p.skeleton || p.ignorePreZ))) {
                                d.multiply(v.WORLDVIEWPROJECTION, v.VIEWPROJECTION, _), p.cullFace !== o && (o = p.cullFace, n.cullFace(o)), p.frontFace !== s && (s = p.frontFace, n.frontFace(s)), p.culling !== a && (a = p.culling, a ? n.enable(n.CULL_FACE) : n.disable(n.CULL_FACE));
                                var g = h.matrixSemantics.WORLDVIEWPROJECTION;
                                h.setUniform(n, g.type, g.symbol, v.WORLDVIEWPROJECTION), p.render(n, i.shader)
                            }
                        }
                    }
                    return n.depthFunc(n.LEQUAL), n.colorMask(!0, !0, !0, !0), n.depthMask(!0), c
                },
                isFrustumCulled: function() {
                    var e = new o,
                        t = new s;
                    return function(r, n, i, a, o) {
                        var s = r.boundingBox || r.geometry.boundingBox;
                        if (t._array = a, e.copy(s), e.applyTransform(t), n && r.isRenderable() && r.castShadow && n.viewBoundingBoxLastFrame.union(e), r.frustumCulling) {
                            if (!e.intersectBoundingBox(i.frustum.boundingBox)) return !0;
                            t._array = o, e.max._array[2] > 0 && e.min._array[2] < 0 && (e.max._array[2] = -1e-20), e.applyProjection(t);
                            var u = e.min._array,
                                h = e.max._array;
                            if (h[0] < -1 || u[0] > 1 || h[1] < -1 || u[1] > 1 || h[2] < -1 || u[2] > 1) return !0
                        }
                        return !1
                    }
                }(),
                disposeScene: function(e) { this.disposeNode(e, !0, !0), e.dispose() },
                disposeNode: function(e, t, r) {
                    var n = {},
                        i = this.gl;
                    e.getParent() && e.getParent().remove(e), e.traverse(function(e) { e.geometry && t && e.geometry.dispose(i), e.material && (n[e.material.__GUID__] = e.material), e.dispose && e.dispose(i) });
                    for (var a in n) { n[a].dispose(i, r) }
                },
                disposeShader: function(e) { e.dispose(this.gl) },
                disposeGeometry: function(e) { e.dispose(this.gl) },
                disposeTexture: function(e) { e.dispose(this.gl) },
                disposeFrameBuffer: function(e) { e.dispose(this.gl) },
                dispose: function() { i.dispose(this.gl) },
                screenToNDC: function(e, t, r) {
                    r || (r = new h), t = this._height - t;
                    var n = this.viewport,
                        i = r._array;
                    return i[0] = (e - n.x) / n.width, i[0] = 2 * i[0] - 1, i[1] = (t - n.y) / n.height, i[1] = 2 * i[1] - 1, r
                }
            });
        g.opaqueSortFunc = g.prototype.opaqueSortFunc = function(e, t) { return e.renderOrder === t.renderOrder ? e.material.shader === t.material.shader ? e.material === t.material ? e.geometry.__GUID__ - t.geometry.__GUID__ : e.material.__GUID__ - t.material.__GUID__ : e.material.shader.__GUID__ - t.material.shader.__GUID__ : e.renderOrder - t.renderOrder }, g.transparentSortFunc = g.prototype.transparentSortFunc = function(e, t) { return e.renderOrder === t.renderOrder ? e.__depth === t.__depth ? e.material.shader === t.material.shader ? e.material === t.material ? e.geometry.__GUID__ - t.geometry.__GUID__ : e.material.__GUID__ - t.material.__GUID__ : e.material.shader.__GUID__ - t.material.shader.__GUID__ : e.__depth - t.__depth : e.renderOrder - t.renderOrder };
        var v = { WORLD: p(), VIEW: p(), PROJECTION: p(), WORLDVIEW: p(), VIEWPROJECTION: p(), WORLDVIEWPROJECTION: p(), WORLDINVERSE: p(), VIEWINVERSE: p(), PROJECTIONINVERSE: p(), WORLDVIEWINVERSE: p(), VIEWPROJECTIONINVERSE: p(), WORLDVIEWPROJECTIONINVERSE: p(), WORLDTRANSPOSE: p(), VIEWTRANSPOSE: p(), PROJECTIONTRANSPOSE: p(), WORLDVIEWTRANSPOSE: p(), VIEWPROJECTIONTRANSPOSE: p(), WORLDVIEWPROJECTIONTRANSPOSE: p(), WORLDINVERSETRANSPOSE: p(), VIEWINVERSETRANSPOSE: p(), PROJECTIONINVERSETRANSPOSE: p(), WORLDVIEWINVERSETRANSPOSE: p(), VIEWPROJECTIONINVERSETRANSPOSE: p(), WORLDVIEWPROJECTIONINVERSETRANSPOSE: p() };
        g.COLOR_BUFFER_BIT = a.COLOR_BUFFER_BIT, g.DEPTH_BUFFER_BIT = a.DEPTH_BUFFER_BIT, g.STENCIL_BUFFER_BIT = a.STENCIL_BUFFER_BIT, e.exports = g
    }, function(e, t) {
        function r(e, t) { this.action = e, this.context = t }
        var n = {
            trigger: function(e) {
                if (this.hasOwnProperty("__handlers__") && this.__handlers__.hasOwnProperty(e)) {
                    var t = this.__handlers__[e],
                        r = t.length,
                        n = -1,
                        i = arguments;
                    switch (i.length) {
                        case 1:
                            for (; ++n < r;) t[n].action.call(t[n].context);
                            return;
                        case 2:
                            for (; ++n < r;) t[n].action.call(t[n].context, i[1]);
                            return;
                        case 3:
                            for (; ++n < r;) t[n].action.call(t[n].context, i[1], i[2]);
                            return;
                        case 4:
                            for (; ++n < r;) t[n].action.call(t[n].context, i[1], i[2], i[3]);
                            return;
                        case 5:
                            for (; ++n < r;) t[n].action.call(t[n].context, i[1], i[2], i[3], i[4]);
                            return;
                        default:
                            for (; ++n < r;) t[n].action.apply(t[n].context, Array.prototype.slice.call(i, 1));
                            return
                    }
                }
            },
            on: function(e, t, n) { if (e && t) { var i = this.__handlers__ || (this.__handlers__ = {}); if (i[e]) { if (this.has(e, t)) return } else i[e] = []; var a = new r(t, n || this); return i[e].push(a), this } },
            once: function(e, t, r) {
                function n() { i.off(e, n), t.apply(this, arguments) }
                if (e && t) { var i = this; return this.on(e, n, r) }
            },
            before: function(e, t, r) { if (e && t) return e = "before" + e, this.on(e, t, r) },
            after: function(e, t, r) { if (e && t) return e = "after" + e, this.on(e, t, r) },
            success: function(e, t) { return this.once("success", e, t) },
            error: function(e, t) { return this.once("error", e, t) },
            off: function(e, t) {
                var r = this.__handlers__ || (this.__handlers__ = {});
                if (!t) return void(r[e] = []);
                if (r[e]) {
                    for (var n = r[e], i = [], a = 0; a < n.length; a++) t && n[a].action !== t && i.push(n[a]);
                    r[e] = i
                }
                return this
            },
            has: function(e, t) {
                var r = this.__handlers__;
                if (!r || !r[e]) return !1;
                for (var n = r[e], i = 0; i < n.length; i++)
                    if (n[i].action === t) return !0
            }
        };
        e.exports = n
    }, function(e, t, r) {
        "use strict";
        var n = (r(3), r(14)),
            i = r(79),
            a = r(1),
            o = a.vec3,
            s = o.set,
            u = o.copy,
            h = o.transformMat4,
            l = Math.min,
            c = Math.max,
            d = function() {
                this.planes = [];
                for (var e = 0; e < 6; e++) this.planes.push(new i);
                this.boundingBox = new n, this.vertices = [];
                for (var e = 0; e < 8; e++) this.vertices[e] = o.fromValues(0, 0, 0)
            };
        d.prototype = {
            setFromProjection: function(e) {
                var t = this.planes,
                    r = e._array,
                    n = r[0],
                    i = r[1],
                    a = r[2],
                    o = r[3],
                    u = r[4],
                    h = r[5],
                    l = r[6],
                    c = r[7],
                    d = r[8],
                    f = r[9],
                    p = r[10],
                    _ = r[11],
                    m = r[12],
                    g = r[13],
                    v = r[14],
                    y = r[15];
                s(t[0].normal._array, o - n, c - u, _ - d), t[0].distance = -(y - m), t[0].normalize(), s(t[1].normal._array, o + n, c + u, _ + d), t[1].distance = -(y + m), t[1].normalize(), s(t[2].normal._array, o + i, c + h, _ + f), t[2].distance = -(y + g), t[2].normalize(), s(t[3].normal._array, o - i, c - h, _ - f), t[3].distance = -(y - g), t[3].normalize(), s(t[4].normal._array, o - a, c - l, _ - p), t[4].distance = -(y - v), t[4].normalize(), s(t[5].normal._array, o + a, c + l, _ + p), t[5].distance = -(y + v), t[5].normalize();
                var x = this.boundingBox;
                if (0 === y) {
                    var T = h / n,
                        b = -v / (p - 1),
                        w = -v / (p + 1),
                        E = -w / h,
                        S = -b / h;
                    x.min.set(-E * T, -E, w), x.max.set(E * T, E, b);
                    var A = this.vertices;
                    s(A[0], -E * T, -E, w), s(A[1], -E * T, E, w), s(A[2], E * T, -E, w), s(A[3], E * T, E, w), s(A[4], -S * T, -S, b), s(A[5], -S * T, S, b), s(A[6], S * T, -S, b), s(A[7], S * T, S, b)
                } else {
                    var M = (-1 - m) / n,
                        N = (1 - m) / n,
                        C = (1 - g) / h,
                        L = (-1 - g) / h,
                        D = (-1 - v) / p,
                        I = (1 - v) / p;
                    x.min.set(Math.min(M, N), Math.min(L, C), Math.min(I, D)), x.max.set(Math.max(N, M), Math.max(C, L), Math.max(D, I));
                    var R = x.min._array,
                        P = x.max._array,
                        A = this.vertices;
                    s(A[0], R[0], R[1], R[2]), s(A[1], R[0], P[1], R[2]), s(A[2], P[0], R[1], R[2]), s(A[3], P[0], P[1], R[2]), s(A[4], R[0], R[1], P[2]), s(A[5], R[0], P[1], P[2]), s(A[6], P[0], R[1], P[2]), s(A[7], P[0], P[1], P[2])
                }
            },
            getTransformedBoundingBox: function() {
                var e = o.create();
                return function(t, r) {
                    var n = this.vertices,
                        i = r._array,
                        a = t.min,
                        o = t.max,
                        s = a._array,
                        d = o._array,
                        f = n[0];
                    h(e, f, i), u(s, e), u(d, e);
                    for (var p = 1; p < 8; p++) f = n[p], h(e, f, i), s[0] = l(e[0], s[0]), s[1] = l(e[1], s[1]), s[2] = l(e[2], s[2]), d[0] = c(e[0], d[0]), d[1] = c(e[1], d[1]), d[2] = c(e[2], d[2]);
                    return a._dirty = !0, o._dirty = !0, t
                }
            }()
        }, e.exports = d
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.quat,
            a = function(e, t, r, n) { e = e || 0, t = t || 0, r = r || 0, n = void 0 === n ? 1 : n, this._array = i.fromValues(e, t, r, n), this._dirty = !0 };
        a.prototype = {
            constructor: a,
            add: function(e) { return i.add(this._array, this._array, e._array), this._dirty = !0, this },
            calculateW: function() { return i.calculateW(this._array, this._array), this._dirty = !0, this },
            set: function(e, t, r, n) { return this._array[0] = e, this._array[1] = t, this._array[2] = r, this._array[3] = n, this._dirty = !0, this },
            setArray: function(e) { return this._array[0] = e[0], this._array[1] = e[1], this._array[2] = e[2], this._array[3] = e[3], this._dirty = !0, this },
            clone: function() { return new a(this.x, this.y, this.z, this.w) },
            conjugate: function() { return i.conjugate(this._array, this._array), this._dirty = !0, this },
            copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this },
            dot: function(e) { return i.dot(this._array, e._array) },
            fromMat3: function(e) { return i.fromMat3(this._array, e._array), this._dirty = !0, this },
            fromMat4: function() {
                var e = n.mat3,
                    t = e.create();
                return function(r) { return e.fromMat4(t, r._array), e.transpose(t, t), i.fromMat3(this._array, t), this._dirty = !0, this }
            }(),
            identity: function() { return i.identity(this._array), this._dirty = !0, this },
            invert: function() { return i.invert(this._array, this._array), this._dirty = !0, this },
            len: function() { return i.len(this._array) },
            length: function() { return i.length(this._array) },
            lerp: function(e, t, r) { return i.lerp(this._array, e._array, t._array, r), this._dirty = !0, this },
            mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this },
            mulLeft: function(e) { return i.multiply(this._array, e._array, this._array), this._dirty = !0, this },
            multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this },
            multiplyLeft: function(e) { return i.multiply(this._array, e._array, this._array), this._dirty = !0, this },
            normalize: function() { return i.normalize(this._array, this._array), this._dirty = !0, this },
            rotateX: function(e) { return i.rotateX(this._array, this._array, e), this._dirty = !0, this },
            rotateY: function(e) { return i.rotateY(this._array, this._array, e), this._dirty = !0, this },
            rotateZ: function(e) { return i.rotateZ(this._array, this._array, e), this._dirty = !0, this },
            rotationTo: function(e, t) { return i.rotationTo(this._array, e._array, t._array), this._dirty = !0, this },
            setAxes: function(e, t, r) { return i.setAxes(this._array, e._array, t._array, r._array), this._dirty = !0, this },
            setAxisAngle: function(e, t) { return i.setAxisAngle(this._array, e._array, t), this._dirty = !0, this },
            slerp: function(e, t, r) { return i.slerp(this._array, e._array, t._array, r), this._dirty = !0, this },
            sqrLen: function() { return i.sqrLen(this._array) },
            squaredLength: function() { return i.squaredLength(this._array) },
            fromEuler: function(e, t) { return a.fromEuler(this, e, t) },
            toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" },
            toArray: function() { return Array.prototype.slice.call(this._array) }
        };
        var o = Object.defineProperty;
        if (o) {
            var s = a.prototype;
            o(s, "x", { get: function() { return this._array[0] }, set: function(e) { this._array[0] = e, this._dirty = !0 } }), o(s, "y", { get: function() { return this._array[1] }, set: function(e) { this._array[1] = e, this._dirty = !0 } }), o(s, "z", { get: function() { return this._array[2] }, set: function(e) { this._array[2] = e, this._dirty = !0 } }), o(s, "w", { get: function() { return this._array[3] }, set: function(e) { this._array[3] = e, this._dirty = !0 } })
        }
        a.add = function(e, t, r) { return i.add(e._array, t._array, r._array), e._dirty = !0, e }, a.set = function(e, t, r, n, a) { i.set(e._array, t, r, n, a), e._dirty = !0 }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.calculateW = function(e, t) { return i.calculateW(e._array, t._array), e._dirty = !0, e }, a.conjugate = function(e, t) { return i.conjugate(e._array, t._array), e._dirty = !0, e }, a.identity = function(e) { return i.identity(e._array), e._dirty = !0, e }, a.invert = function(e, t) { return i.invert(e._array, t._array), e._dirty = !0, e }, a.dot = function(e, t) { return i.dot(e._array, t._array) }, a.len = function(e) { return i.length(e._array) }, a.lerp = function(e, t, r, n) { return i.lerp(e._array, t._array, r._array, n), e._dirty = !0, e }, a.slerp = function(e, t, r, n) { return i.slerp(e._array, t._array, r._array, n), e._dirty = !0, e }, a.mul = function(e, t, r) { return i.multiply(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.rotateX = function(e, t, r) { return i.rotateX(e._array, t._array, r), e._dirty = !0, e }, a.rotateY = function(e, t, r) { return i.rotateY(e._array, t._array, r), e._dirty = !0, e }, a.rotateZ = function(e, t, r) { return i.rotateZ(e._array, t._array, r), e._dirty = !0, e }, a.setAxisAngle = function(e, t, r) { return i.setAxisAngle(e._array, t._array, r), e._dirty = !0, e }, a.normalize = function(e, t) { return i.normalize(e._array, t._array), e._dirty = !0, e }, a.sqrLen = function(e) { return i.sqrLen(e._array) }, a.squaredLength = a.sqrLen, a.fromMat3 = function(e, t) { return i.fromMat3(e._array, t._array), e._dirty = !0, e }, a.setAxes = function(e, t, r, n) { return i.setAxes(e._array, t._array, r._array, n._array), e._dirty = !0, e }, a.rotationTo = function(e, t, r) { return i.rotationTo(e._array, t._array, r._array), e._dirty = !0, e }, a.fromEuler = function(e, t, r) {
            e._dirty = !0, t = t._array;
            var n = e._array,
                i = Math.cos(t[0] / 2),
                a = Math.cos(t[1] / 2),
                o = Math.cos(t[2] / 2),
                s = Math.sin(t[0] / 2),
                u = Math.sin(t[1] / 2),
                h = Math.sin(t[2] / 2),
                r = (r || "XYZ").toUpperCase();
            switch (r) {
                case "XYZ":
                    n[0] = s * a * o + i * u * h, n[1] = i * u * o - s * a * h, n[2] = i * a * h + s * u * o, n[3] = i * a * o - s * u * h;
                    break;
                case "YXZ":
                    n[0] = s * a * o + i * u * h, n[1] = i * u * o - s * a * h, n[2] = i * a * h - s * u * o, n[3] = i * a * o + s * u * h;
                    break;
                case "ZXY":
                    n[0] = s * a * o - i * u * h, n[1] = i * u * o + s * a * h, n[2] = i * a * h + s * u * o, n[3] = i * a * o - s * u * h;
                    break;
                case "ZYX":
                    n[0] = s * a * o - i * u * h, n[1] = i * u * o + s * a * h, n[2] = i * a * h - s * u * o, n[3] = i * a * o + s * u * h;
                    break;
                case "YZX":
                    n[0] = s * a * o + i * u * h, n[1] = i * u * o + s * a * h, n[2] = i * a * h - s * u * o, n[3] = i * a * o - s * u * h;
                    break;
                case "XZY":
                    n[0] = s * a * o - i * u * h, n[1] = i * u * o - s * a * h, n[2] = i * a * h + s * u * o, n[3] = i * a * o + s * u * h
            }
        }, e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(3),
            i = r(1),
            a = i.vec3,
            o = function(e, t) { this.origin = e || new n, this.direction = t || new n };
        o.prototype = {
            constructor: o,
            intersectPlane: function(e, t) {
                var r = e.normal._array,
                    i = e.distance,
                    o = this.origin._array,
                    s = this.direction._array,
                    u = a.dot(r, s);
                if (0 === u) return null;
                t || (t = new n);
                var h = (a.dot(r, o) - i) / u;
                return a.scaleAndAdd(t._array, o, s, -h), t._dirty = !0, t
            },
            mirrorAgainstPlane: function(e) {
                var t = a.dot(e.normal._array, this.direction._array);
                a.scaleAndAdd(this.direction._array, this.direction._array, e.normal._array, 2 * -t), this.direction._dirty = !0
            },
            distanceToPoint: function() { var e = a.create(); return function(t) { a.sub(e, t, this.origin._array); var r = a.dot(e, this.direction._array); if (r < 0) return a.distance(this.origin._array, t); var n = a.lenSquared(e); return Math.sqrt(n - r * r) } }(),
            intersectSphere: function() {
                var e = a.create();
                return function(t, r, i) {
                    var o = this.origin._array,
                        s = this.direction._array;
                    t = t._array, a.sub(e, t, o);
                    var u = a.dot(e, s),
                        h = a.squaredLength(e),
                        l = h - u * u,
                        c = r * r;
                    if (!(l > c)) {
                        var d = Math.sqrt(c - l),
                            f = u - d,
                            p = u + d;
                        return i || (i = new n), f < 0 ? p < 0 ? null : (a.scaleAndAdd(i._array, o, s, p), i) : (a.scaleAndAdd(i._array, o, s, f), i)
                    }
                }
            }(),
            intersectBoundingBox: function(e, t) {
                var r, i, o, s, u, h, l = this.direction._array,
                    c = this.origin._array,
                    d = e.min._array,
                    f = e.max._array,
                    p = 1 / l[0],
                    _ = 1 / l[1],
                    m = 1 / l[2];
                if (p >= 0 ? (r = (d[0] - c[0]) * p, i = (f[0] - c[0]) * p) : (i = (d[0] - c[0]) * p, r = (f[0] - c[0]) * p), _ >= 0 ? (o = (d[1] - c[1]) * _, s = (f[1] - c[1]) * _) : (s = (d[1] - c[1]) * _, o = (f[1] - c[1]) * _), r > s || o > i) return null;
                if ((o > r || r !== r) && (r = o), (s < i || i !== i) && (i = s), m >= 0 ? (u = (d[2] - c[2]) * m, h = (f[2] - c[2]) * m) : (h = (d[2] - c[2]) * m, u = (f[2] - c[2]) * m), r > h || u > i) return null;
                if ((u > r || r !== r) && (r = u), (h < i || i !== i) && (i = h), i < 0) return null;
                var g = r >= 0 ? r : i;
                return t || (t = new n), a.scaleAndAdd(t._array, c, l, g), t
            },
            intersectTriangle: function() {
                var e = a.create(),
                    t = a.create(),
                    r = a.create(),
                    i = a.create();
                return function(o, s, u, h, l, c) {
                    var d = this.direction._array,
                        f = this.origin._array;
                    o = o._array, s = s._array, u = u._array, a.sub(e, s, o), a.sub(t, u, o), a.cross(i, t, d);
                    var p = a.dot(e, i);
                    if (h) { if (p > -1e-5) return null } else if (p > -1e-5 && p < 1e-5) return null;
                    a.sub(r, f, o);
                    var _ = a.dot(i, r) / p;
                    if (_ < 0 || _ > 1) return null;
                    a.cross(i, e, r);
                    var m = a.dot(d, i) / p;
                    if (m < 0 || m > 1 || _ + m > 1) return null;
                    a.cross(i, e, t);
                    var g = -a.dot(r, i) / p;
                    return g < 0 ? null : (l || (l = new n), c && n.set(c, 1 - _ - m, _, m), a.scaleAndAdd(l._array, f, d, g), l)
                }
            }(),
            applyTransform: function(e) { n.add(this.direction, this.direction, this.origin), n.transformMat4(this.origin, this.origin, e), n.transformMat4(this.direction, this.direction, e), n.sub(this.direction, this.direction, this.origin), n.normalize(this.direction, this.direction) },
            copy: function(e) { n.copy(this.origin, e.origin), n.copy(this.direction, e.direction) },
            clone: function() { var e = new o; return e.copy(this), e }
        }, e.exports = o
    }, function(e, t, r) {
        var n = r(25),
            i = r(74),
            a = r(7),
            o = r(16);
        a.import(r(226));
        var s = n.extend(function() {
            var e = new a({ vertex: a.source("qtek.skybox.vertex"), fragment: a.source("qtek.skybox.fragment") }),
                t = new o({ shader: e, depthMask: !1 });
            return { scene: null, geometry: new i, material: t, environmentMap: null, culling: !1 }
        }, function() {
            var e = this.scene;
            e && this.attachScene(e), this.environmentMap && this.setEnvironmentMap(this.environmentMap)
        }, { attachScene: function(e) { this.scene && this.detachScene(), this.scene = e, e.on("beforerender", this._beforeRenderScene, this) }, detachScene: function() { this.scene && this.scene.off("beforerender", this._beforeRenderScene), this.scene = null }, dispose: function(e) { this.detachScene(), this.geometry.dispose(e), this.material.dispose(e) }, setEnvironmentMap: function(e) { this.material.set("environmentMap", e) }, getEnvironmentMap: function() { return this.material.get("environmentMap") }, _beforeRenderScene: function(e, t, r) { this.renderSkybox(e, r) }, renderSkybox: function(e, t) { this.position.copy(t.getWorldPosition()), this.update(), e.gl.disable(e.gl.BLEND), e.renderQueue([this], t) } });
        e.exports = s
    }, function(e, t, r) {
        var n = r(25),
            i = r(75),
            a = r(7),
            o = r(16);
        a.import(r(212));
        var s = n.extend(function() {
            var e = new a({ vertex: a.source("qtek.basic.vertex"), fragment: a.source("qtek.basic.fragment") });
            e.enableTexture("diffuseMap");
            var t = new o({ shader: e, depthMask: !1 });
            return { scene: null, geometry: new i({ widthSegments: 30, heightSegments: 30 }), material: t, environmentMap: null, culling: !1 }
        }, function() {
            var e = this.scene;
            e && this.attachScene(e), this.environmentMap && this.setEnvironmentMap(this.environmentMap)
        }, { attachScene: function(e) { this.scene && this.detachScene(), this.scene = e, e.on("beforerender", this._beforeRenderScene, this) }, detachScene: function() { this.scene && this.scene.off("beforerender", this._beforeRenderScene), this.scene = null }, _beforeRenderScene: function(e, t, r) { this.position.copy(r.getWorldPosition()), this.update(), e.renderQueue([this], r) }, setEnvironmentMap: function(e) { this.material.set("diffuseMap", e) }, getEnvironmentMap: function() { return this.material.get("diffuseMap") }, dispose: function(e) { this.detachScene(), this.geometry.dispose(e), this.material.dispose(e) } });
        e.exports = s
    }, function(e, t, r) {
        var n = r(8),
            i = r(3),
            a = r(44),
            o = r(10),
            s = ["px", "nx", "py", "ny", "pz", "nz"],
            u = n.extend(function() {
                var e = { position: new i, far: 1e3, near: .1, texture: null, shadowMapPass: null },
                    t = e._cameras = { px: new a({ fov: 90 }), nx: new a({ fov: 90 }), py: new a({ fov: 90 }), ny: new a({ fov: 90 }), pz: new a({ fov: 90 }), nz: new a({ fov: 90 }) };
                return t.px.lookAt(i.POSITIVE_X, i.NEGATIVE_Y), t.nx.lookAt(i.NEGATIVE_X, i.NEGATIVE_Y), t.py.lookAt(i.POSITIVE_Y, i.POSITIVE_Z), t.ny.lookAt(i.NEGATIVE_Y, i.NEGATIVE_Z), t.pz.lookAt(i.POSITIVE_Z, i.NEGATIVE_Y), t.nz.lookAt(i.NEGATIVE_Z, i.NEGATIVE_Y), e._frameBuffer = new o, e
            }, {
                getCamera: function(e) { return this._cameras[e] },
                render: function(e, t, r) {
                    var n = e.gl;
                    r || t.update();
                    for (var a = this.texture.width, o = 2 * Math.atan(a / (a - .5)) / Math.PI * 180, u = 0; u < 6; u++) {
                        var h = s[u],
                            l = this._cameras[h];
                        if (i.copy(l.position, this.position), l.far = this.far, l.near = this.near, l.fov = o, this.shadowMapPass) {
                            l.update();
                            var c = t.getBoundingBox(function(e) { return !e.invisible });
                            c.applyTransform(l.viewMatrix), t.viewBoundingBoxLastFrame.copy(c), this.shadowMapPass.render(e, t, l, !0)
                        }
                        this._frameBuffer.attach(this.texture, n.COLOR_ATTACHMENT0, n.TEXTURE_CUBE_MAP_POSITIVE_X + u), this._frameBuffer.bind(e), e.render(t, l, !0), this._frameBuffer.unbind(e)
                    }
                },
                dispose: function(e) { this._frameBuffer.dispose(e) }
            });
        e.exports = u
    }, function(e, t) {
        var r = function() { this.head = null, this.tail = null, this._len = 0 },
            n = r.prototype;
        n.insert = function(e) { var t = new i(e); return this.insertEntry(t), t }, n.insertEntry = function(e) { this.head ? (this.tail.next = e, e.prev = this.tail, e.next = null, this.tail = e) : this.head = this.tail = e, this._len++ }, n.remove = function(e) {
            var t = e.prev,
                r = e.next;
            t ? t.next = r : this.head = r, r ? r.prev = t : this.tail = t, e.next = e.prev = null, this._len--
        }, n.len = function() { return this._len }, n.clear = function() { this.head = this.tail = null, this._len = 0 };
        var i = function(e) { this.value = e, this.next, this.prev },
            a = function(e) { this._list = new r, this._map = {}, this._maxSize = e || 10, this._lastRemovedEntry = null },
            o = a.prototype;
        o.put = function(e, t) {
            var r = this._list,
                n = this._map,
                a = null;
            if (null == n[e]) {
                var o = r.len(),
                    s = this._lastRemovedEntry;
                if (o >= this._maxSize && o > 0) {
                    var u = r.head;
                    r.remove(u), delete n[u.key], a = u.value, this._lastRemovedEntry = u
                }
                s ? s.value = t : s = new i(t), s.key = e, r.insertEntry(s), n[e] = s
            }
            return a
        }, o.get = function(e) {
            var t = this._map[e],
                r = this._list;
            if (null != t) return t !== r.tail && (r.remove(t), r.insertEntry(t)), t.value
        }, o.clear = function() { this._list.clear(), this._map = {} }, e.exports = a
    }, function(e, t, r) {
        function n(e) {
            this.rootNode = new a.Node, this._currentMap = "", this._triangulationResults = {}, this._shadersMap = a.COMMON_SHADERS.reduce(function(e, t) { return e[t] = a.createShader("ecgl." + t), e[t].define("fragment", "DOUBLE_SIDED"), e }, {}), this._linesShader = a.createShader("ecgl.meshLines3D");
            var t = {};
            a.COMMON_SHADERS.forEach(function(e) { t[e] = new a.Material({ shader: a.createShader("ecgl." + e) }) }), this._groundMaterials = t, this._groundMesh = new a.Mesh({ geometry: new a.PlaneGeometry({ dynamic: !0 }), castShadow: !1, renderNormal: !0, $ignorePicking: !0 }), this._groundMesh.rotation.rotateX(-Math.PI / 2), this._labelsBuilder = new c(1024, 1024, e), this._labelsBuilder.getMesh().renderOrder = 100, this._labelsBuilder.getMesh().material.depthTest = !1, this._api = e
        }
        var i = r(0),
            a = r(2),
            o = r(172),
            s = r(22),
            u = r(4),
            h = r(1),
            l = r(50),
            c = r(49),
            d = h.vec3;
        a.Shader.import(r(41)), n.prototype = {
            constructor: n,
            extrudeY: !0,
            update: function(e, t, r, n) {
                var i = e.get("instancing");
                this._triangulation(t), (t.map !== this._currentMap || i && !this._polygonMesh || !i && !this._polygonMeshesMap) && (this._currentMap = t.map, this._initMeshes(e, t), this.rootNode.add(this._labelsBuilder.getMesh()));
                var a = this._getShader(e.get("shading")),
                    o = e.getData();
                i && this._prepareInstancingMesh(e, t, a, n), this.rootNode.updateWorldTransform(), this._updateRegionMesh(e, t, a, n, i), this._updateGroundPlane(e, t, n), this._labelsBuilder.updateData(o), this._labelsBuilder.getLabelPosition = function(e, r, n) {
                    var i = o.getName(e),
                        a = t.getRegion(i),
                        s = a.center,
                        u = n;
                    return t.dataToPoint([s[0], s[1], u])
                }, this._data = o, this._labelsBuilder.updateLabels(), this._updateDebugWireframe(e, t), this._lastHoverDataIndex = 0
            },
            _prepareInstancingMesh: function(e, t, r, n) {
                var i = 0,
                    a = 0;
                t.regions.forEach(function(e) {
                    var t = this._getRegionPolygonGeoInfo(e);
                    i += t.vertexCount, a += t.triangleCount
                }, this);
                var o = this._polygonMesh,
                    s = o.geometry;
                ["position", "normal", "texcoord0", "color"].forEach(function(e) { s.attributes[e].init(i) }), s.indices = i > 65535 ? new Uint32Array(3 * a) : new Uint16Array(3 * a), o.material.shader !== r && o.material.attachShader(r, !0), this._dataIndexOfVertex = new Uint32Array(i), this._vertexRangeOfDataIndex = new Uint32Array(2 * t.regions.length)
            },
            _updateRegionMesh: function(e, t, r, n, i) {
                var o = e.getData(),
                    s = 0,
                    h = 0;
                i && a.setMaterialFromModel(r.__shading, this._polygonMesh.material, e, n);
                var l = !1,
                    c = {};
                if (o.each(function(e) { c[o.getName(e)] = e }), t.regions.forEach(function(d) {
                        var f = c[d.name];
                        null == f && (f = -1);
                        var p = i ? this._polygonMesh : this._polygonMeshesMap[d.name],
                            _ = i ? this._linesMesh : this._linesMeshesMap[d.name];
                        p.material.shader !== r && p.material.attachShader(r, !0);
                        var m = e.getRegionModel(d.name),
                            g = m.getModel("itemStyle"),
                            v = g.get("areaColor"),
                            y = u.firstNotNull(g.get("opacity"), 1),
                            x = o.getItemVisual(f, "color", !0);
                        null != x && o.hasValue(f) && (v = x), o.setItemVisual(f, "color", v), o.setItemVisual(f, "opacity", y), v = a.parseColor(v);
                        var T = a.parseColor(g.get("borderColor"));
                        v[3] *= y, T[3] *= y;
                        var b = v[3] < .99;
                        i ? p.material.set("color", [1, 1, 1, 1]) : (a.setMaterialFromModel(r.__shading, p.material, m, n), p.material.set({ color: v }), p.material.transparent = b, p.material.depthMask = !b), l = l || b;
                        var w = u.firstNotNull(m.get("height", !0), e.get("regionHeight"));
                        if (i) {
                            for (var E = this._updatePolygonGeometry(e, p.geometry, d, w, s, h, v), S = s; S < E.vertexOffset; S++) this._dataIndexOfVertex[S] = f;
                            this._vertexRangeOfDataIndex[2 * f] = s, this._vertexRangeOfDataIndex[2 * f + 1] = E.vertexOffset, s = E.vertexOffset, h = E.triangleOffset
                        } else this._updatePolygonGeometry(e, p.geometry, d, w);
                        var A = g.get("borderWidth"),
                            M = A > 0;
                        i || (M && (A *= n.getDevicePixelRatio(), this._updateLinesGeometry(_.geometry, d, w, A, t.transform)), _.invisible = !M, _.material.set({ color: T })), i || (this._moveRegionToCenter(p, _, M), "geo3D" === e.type ? p.eventData = { name: d.name } : (p.dataIndex = f, p.seriesIndex = e.seriesIndex), p.on("mouseover", this._onmouseover, this), p.on("mouseout", this._onmouseout, this), p.material.get("normalMap") && p.geometry.generateTangents())
                    }, this), i) {
                    var d = this._polygonMesh;
                    d.material.transparent = l, d.material.depthMask = !l, d.geometry.updateBoundingBox(), d.material.get("normalMap") && d.geometry.generateTangents(), d.seriesIndex = e.seriesIndex, d.on("mousemove", this._onmousemove, this), d.on("mouseout", this._onmouseout, this)
                }
            },
            _updateDebugWireframe: function(e, t) {
                var r = e.getModel("debug.wireframe");
                if (r.get("show")) {
                    var n = a.parseColor(r.get("lineStyle.color") || "rgba(0,0,0,0.5)"),
                        i = u.firstNotNull(r.get("lineStyle.width"), 1),
                        o = function(e) { e.geometry.generateBarycentric(), e.material.shader.define("both", "WIREFRAME_TRIANGLE"), e.material.set("wireframeLineColor", n), e.material.set("wireframeLineWidth", i) };
                    this._polygonMeshesMap ? t.regions.forEach(function(e) { o(this._polygonMeshesMap[e.name]) }, this) : o(this._polygonMesh)
                }
            },
            _onmousemove: function(e) {
                var t = this._dataIndexOfVertex[e.triangle[0]];
                null == t && (t = -1), t !== this._lastHoverDataIndex && (this.downplay(this._lastHoverDataIndex), this.highlight(t)), this._lastHoverDataIndex = t, this._polygonMesh.dataIndex = t
            },
            _onmouseover: function(e) {
                if (e.target) {
                    var t = e.target.eventData ? this._data.indexOfName(e.target.eventData.name) : e.target.dataIndex;
                    null != t && (this.highlight(t), this._labelsBuilder.updateLabels([t]))
                }
            },
            _onmouseout: function(e) {
                if (e.target)
                    if (this._polygonMesh) this.downplay(this._lastHoverDataIndex), this._lastHoverDataIndex = -1, this._polygonMesh.dataIndex = -1;
                    else {
                        var t = e.target.eventData ? this._data.indexOfName(e.target.eventData.name) : e.target.dataIndex;
                        null != t && (this.downplay(t), e.relatedTarget || this._labelsBuilder.updateLabels())
                    }
            },
            _updateGroundPlane: function(e, t, r) {
                var n = e.getModel("groundPlane", e);
                if (this._groundMesh.invisible = !n.get("show", !0), !this._groundMesh.invisible) {
                    var i = e.get("shading"),
                        o = this._groundMaterials[i];
                    o || (o = this._groundMaterials.lambert), a.setMaterialFromModel(i, o, n, r), o.get("normalMap") && this._groundMesh.geometry.generateTangents(), this._groundMesh.material = o, this._groundMesh.material.set("color", a.parseColor(n.get("color"))), this._groundMesh.scale.set(t.size[0], t.size[2], 1)
                }
            },
            _initMeshes: function(e, t) {
                function r() { var e = new a.Mesh({ material: new a.Material({ shader: o }), culling: !1, geometry: new a.Geometry({ sortTriangles: !0, dynamic: !0 }), renderNormal: !0 }); return i.util.extend(e.geometry, l), e }

                function n(e) { return new a.Mesh({ material: new a.Material({ shader: e }), castShadow: !1, $ignorePicking: !0, geometry: new s({ useNativeLine: !1 }) }) }
                this.rootNode.removeAll();
                var o = this._getShader(e.get("shading"));
                if (e.get("instancing")) {
                    var u = r(),
                        h = n(this._linesShader);
                    this.rootNode.add(u), this.rootNode.add(h), u.material.shader.define("both", "VERTEX_COLOR"), this._polygonMesh = u, this._linesMesh = h, this._polygonMeshesMap = null, this._linesMeshesMap = null
                } else {
                    var c = {},
                        d = {};
                    t.regions.forEach(function(e) { c[e.name] = r(), d[e.name] = n(this._linesShader), this.rootNode.add(c[e.name]), this.rootNode.add(d[e.name]) }, this), this._polygonMeshesMap = c, this._linesMeshesMap = d
                }
                this.rootNode.add(this._groundMesh)
            },
            _getShader: function(e) { var t = this._shadersMap[e]; return t || (t = this._shadersMap.lambert), t.__shading = e, t },
            _triangulation: function(e) {
                this._triangulationResults = {};
                var t = [1 / 0, 1 / 0, 1 / 0],
                    r = [-1 / 0, -1 / 0, -1 / 0];
                e.regions.forEach(function(n) {
                    for (var i = [], a = 0; a < n.geometries.length; a++) {
                        var s = n.geometries[a].exterior,
                            u = n.geometries[a].interiors,
                            h = [],
                            l = [];
                        if (!(s.length < 3)) {
                            for (var c = 0, f = 0; f < s.length; f++) {
                                var p = s[f];
                                h[c++] = p[0], h[c++] = p[1]
                            }
                            for (var f = 0; f < u.length; f++)
                                if (!(u[f].length.length < 3)) {
                                    for (var _ = h.length / 2, m = 0; m < u[f].length; m++) {
                                        var p = u[f][m];
                                        h.push(p[0]), h.push(p[1])
                                    }
                                    l.push(_)
                                }
                            for (var g = o(h, l), v = new Float64Array(h.length / 2 * 3), y = [], x = [1 / 0, 1 / 0, 1 / 0], T = [-1 / 0, -1 / 0, -1 / 0], b = 0, f = 0; f < h.length;) d.set(y, h[f++], 0, h[f++]), d.transformMat4(y, y, e.transform), d.min(x, x, y), d.max(T, T, y), v[b++] = y[0], v[b++] = y[1], v[b++] = y[2];
                            d.min(t, t, x), d.max(r, r, T), i.push({ points: v, indices: g })
                        }
                    }
                    this._triangulationResults[n.name] = i
                }, this), this._geoBoundingBox = [t, r]
            },
            _getRegionPolygonGeoInfo: function(e) { for (var t = this._triangulationResults[e.name], r = 0, n = 0, i = 0; i < t.length; i++) r += t[i].points.length / 3, n += t[i].indices.length / 3; return { vertexCount: 2 * r + 4 * r, triangleCount: 2 * n + 2 * r } },
            _updatePolygonGeometry: function(e, t, r, n, i, a, o) {
                function s(e, t, r) { for (var n = e.points, a = n.length, s = [], u = [], h = 0; h < a; h += 3) s[0] = n[h], s[y] = t, s[x] = n[h + 2], u[0] = (n[h] * b[0] - w[0]) / S, u[1] = (n[h + 2] * b[x] - w[2]) / S, l.set(i, s), m && p.set(i, o), f.set(i++, u) }

                function u(e, t, r) {
                    var n = i;
                    s(e, t, r);
                    for (var o = 0; o < e.indices.length; o++) g[3 * a + o] = e.indices[o] + n;
                    a += e.indices.length / 3
                }
                var h = e.get("projectUVOnGround"),
                    l = t.attributes.position,
                    c = t.attributes.normal,
                    f = t.attributes.texcoord0,
                    p = t.attributes.color,
                    _ = this._triangulationResults[r.name],
                    m = p.value && o,
                    g = t.indices,
                    v = null != i,
                    y = this.extrudeY ? 1 : 2,
                    x = this.extrudeY ? 2 : 1;
                if (!v) {
                    var T = this._getRegionPolygonGeoInfo(r);
                    i = a = 0, l.init(T.vertexCount), c.init(T.vertexCount), f.init(T.vertexCount), g = t.indices = T.vertexCount > 65535 ? new Uint32Array(3 * T.triangleCount) : new Uint16Array(3 * T.triangleCount)
                }
                for (var b = [this.rootNode.worldTransform.x.len(), this.rootNode.worldTransform.y.len(), this.rootNode.worldTransform.z.len()], w = d.mul([], this._geoBoundingBox[0], b), E = d.mul([], this._geoBoundingBox[1], b), S = Math.max(E[0] - w[0], E[2] - w[2]), A = this.extrudeY ? [0, 1, 0] : [0, 0, 1], M = d.negate([], A), N = 0; N < _.length; N++) {
                    var C = i,
                        L = _[N];
                    u(L, 0, 0), u(L, n, 0);
                    for (var D = L.points.length / 3, I = 0; I < D; I++) c.set(C + I, M), c.set(C + I + D, A);
                    for (var R = [0, 3, 1, 1, 3, 2], P = [
                            [],
                            [],
                            [],
                            []
                        ], O = [], F = [], B = [], U = [], z = 0, I = 0; I < D; I++) {
                        for (var G = (I + 1) % D, k = (L.points[3 * G] - L.points[3 * I]) * b[0], H = (L.points[3 * G + 2] - L.points[3 * I + 2]) * b[x], V = Math.sqrt(k * k + H * H), W = 0; W < 4; W++) {
                            var q = 0 === W || 3 === W,
                                X = 3 * (q ? I : G);
                            P[W][0] = L.points[X], P[W][y] = W > 1 ? n : 0, P[W][x] = L.points[X + 2], l.set(i + W, P[W]), h ? (U[0] = (L.points[X] * b[0] - w[0]) / S, U[1] = (L.points[X + 2] * b[x] - w[x]) / S) : (U[0] = (q ? z : z + V) / S, U[1] = (P[W][y] * b[y] - w[y]) / S), f.set(i + W, U)
                        }
                        d.sub(O, P[1], P[0]), d.sub(F, P[3], P[0]), d.cross(B, O, F), d.normalize(B, B);
                        for (var W = 0; W < 4; W++) c.set(i + W, B), m && p.set(i + W, o);
                        for (var W = 0; W < 6; W++) g[3 * a + W] = R[W] + i;
                        i += 4, a += 2, z += V
                    }
                }
                return v || t.updateBoundingBox(), t.dirty(), { vertexOffset: i, triangleOffset: a }
            },
            _getRegionLinesGeoInfo: function(e, t) {
                var r = 0,
                    n = 0;
                return e.geometries.forEach(function(e) {
                    var i = e.exterior,
                        a = e.interiors;
                    r += t.getPolylineVertexCount(i), n += t.getPolylineTriangleCount(i);
                    for (var o = 0; o < a.length; o++) r += t.getPolylineVertexCount(a[o]), n += t.getPolylineTriangleCount(a[o])
                }, this), { vertexCount: r, triangleCount: n }
            },
            _updateLinesGeometry: function(e, t, r, n, i) {
                function a(e) { for (var t = new Float64Array(3 * e.length), n = 0, a = [], o = 0; o < e.length; o++) a[0] = e[o][0], a[1] = r + .1, a[2] = e[o][1], d.transformMat4(a, a, i), t[n++] = a[0], t[n++] = a[1], t[n++] = a[2]; return t }
                var o = this._getRegionLinesGeoInfo(t, e);
                e.resetOffset(), e.setVertexCount(o.vertexCount), e.setTriangleCount(o.triangleCount);
                var s = [1, 1, 1, 1];
                t.geometries.forEach(function(t) {
                    var r = t.exterior,
                        i = t.interiors;
                    e.addPolyline(a(r), s, n);
                    for (var o = 0; o < i.length; o++) e.addPolyline(a(i[o]), s, n)
                }), e.updateBoundingBox()
            },
            _moveRegionToCenter: function(e, t, r) {
                var n = e.geometry,
                    i = t.geometry,
                    a = e.geometry.boundingBox,
                    o = a.min.clone().add(a.max).scale(.5),
                    s = o._array;
                a.min.sub(o), a.max.sub(o);
                for (var u = n.attributes.position.value, h = 0; h < u.length;) u[h++] -= s[0], u[h++] -= s[1], u[h++] -= s[2];
                if (e.position.copy(o), r) {
                    i.boundingBox.min.sub(o), i.boundingBox.max.sub(o);
                    for (var l = i.attributes.position.value, h = 0; h < l.length;) l[h++] -= s[0], l[h++] -= s[1], l[h++] -= s[2];
                    t.position.copy(o)
                }
            },
            highlight: function(e) {
                var t = this._data;
                if (t) {
                    var r = t.getItemModel(e),
                        n = r.getModel("emphasis.itemStyle"),
                        o = n.get("areaColor"),
                        s = u.firstNotNull(n.get("opacity"), t.getItemVisual(e, "opacity"), 1);
                    if (null == o) {
                        var h = t.getItemVisual(e, "color");
                        o = i.color.lift(h, -.4)
                    }
                    null == s && (s = t.getItemVisual(e, "opacity"));
                    var l = a.parseColor(o);
                    l[3] *= s, this._setColorOfDataIndex(t, e, l)
                }
            },
            downplay: function(e) {
                var t = this._data;
                if (t) {
                    var r = t.getItemVisual(e, "color"),
                        n = u.firstNotNull(t.getItemVisual(e, "opacity"), 1),
                        i = a.parseColor(r);
                    i[3] *= n, this._setColorOfDataIndex(t, e, i)
                }
            },
            _setColorOfDataIndex: function(e, t, r) {
                if (this._polygonMesh)
                    for (var n = this._vertexRangeOfDataIndex[2 * t]; n < this._vertexRangeOfDataIndex[2 * t + 1]; n++) this._polygonMesh.geometry.attributes.color.set(n, r), this._polygonMesh.geometry.dirty();
                else { var i = this._polygonMeshesMap[e.getName(t)]; if (i) { i.material.set("color", r) } }
                this._api.getZr().refresh()
            }
        }, e.exports = n
    }, function(e, t) { e.exports = function(e, t, r) { var n, i = e.scale; return "ordinal" === i.type && ("function" == typeof r ? (n = i.getTicks()[t], !r(n, i.getLabel(n))) : t % (r + 1)) } }, function(e, t, r) {
        function n(e, t, r, n, i) { this.name = e, this.map = t, this.regionHeight = 0, this.regions = [], this._nameCoordMap = {}, this.loadGeoJson(r, n, i), this.transform = s.identity(new Float64Array(16)), this.invTransform = s.identity(new Float64Array(16)), this.extrudeY = !0, this.altitudeAxis }
        var i = r(0),
            a = r(1),
            o = a.vec3,
            s = a.mat4,
            u = [r(192), r(191)];
        n.prototype = {
            constructor: n,
            type: "geo3D",
            dimensions: ["lng", "lat", "alt"],
            containPoint: function() {},
            loadGeoJson: function(e, t, r) {
                try { this.regions = e ? i.parseGeoJSON(e) : [] } catch (e) { throw "Invalid geoJson format\n" + e }
                t = t || {}, r = r || {};
                for (var n = this.regions, a = {}, o = 0; o < n.length; o++) {
                    var s = n[o].name;
                    s = r[s] || s, n[o].name = s, a[s] = n[o], this.addGeoCoord(s, n[o].center);
                    var h = t[s];
                    h && n[o].transformTo(h.left, h.top, h.width, h.height)
                }
                this._regionsMap = a, this._geoRect = null, u.forEach(function(e) { e(this) }, this)
            },
            getGeoBoundingRect: function() {
                if (this._geoRect) return this._geoRect;
                for (var e, t = this.regions, r = 0; r < t.length; r++) {
                    var n = t[r].getBoundingRect();
                    e = e || n.clone(), e.union(n)
                }
                return this._geoRect = e || new i.graphic.BoundingRect(0, 0, 0, 0)
            },
            addGeoCoord: function(e, t) { this._nameCoordMap[e] = t },
            getRegion: function(e) { return this._regionsMap[e] },
            getRegionByCoord: function(e) {
                for (var t = this.regions, r = 0; r < t.length; r++)
                    if (t[r].contain(e)) return t[r]
            },
            setSize: function(e, t, r) {
                this.size = [e, t, r];
                var n = this.getGeoBoundingRect(),
                    i = e / n.width,
                    a = -r / n.height,
                    o = -e / 2 - n.x * i,
                    u = r / 2 - n.y * a,
                    h = this.extrudeY ? [o, 0, u] : [o, u, 0],
                    l = this.extrudeY ? [i, 1, a] : [i, a, 1],
                    c = this.transform;
                s.identity(c), s.translate(c, c, h), s.scale(c, c, l), s.invert(this.invTransform, c)
            },
            dataToPoint: function(e, t) {
                t = t || [];
                var r = this.extrudeY ? 1 : 2,
                    n = this.extrudeY ? 2 : 1,
                    i = e[2];
                return isNaN(i) && (i = 0), t[0] = e[0], t[n] = e[1], this.altitudeAxis ? t[r] = this.altitudeAxis.dataToCoord(i) : t[r] = 0, t[r] += this.regionHeight, o.transformMat4(t, t, this.transform), t
            },
            pointToData: function(e, t) {}
        }, e.exports = n
    }, function(e, t, r) {
        var n = r(0);
        e.exports = {
            getFilledRegions: function(e, t) {
                var r, i = (e || []).slice();
                if ("string" == typeof t ? (t = n.getMap(t), r = t && t.geoJson) : t && t.features && (r = t), !r) return [];
                for (var a = {}, o = r.features, s = 0; s < i.length; s++) a[i[s].name] = i[s];
                for (var s = 0; s < o.length; s++) {
                    var u = o[s].properties.name;
                    a[u] || i.push({ name: u })
                }
                return i
            },
            defaultOption: { show: !0, zlevel: -10, map: "", left: 0, top: 0, width: "100%", height: "100%", boxWidth: 100, boxHeight: 10, boxDepth: "auto", regionHeight: 3, environment: "auto", groundPlane: { show: !1, color: "#aaa" }, instancing: !1, shading: "lambert", light: { main: { alpha: 40, beta: 30 } }, viewControl: { alpha: 40, beta: 0, distance: 100, orthographicSize: 60, minAlpha: 5, minBeta: -80, maxBeta: 80 }, label: { show: !1, distance: 2, textStyle: { fontSize: 20, color: "#000", backgroundColor: "rgba(255,255,255,0.7)", padding: 3, borderRadius: 4 } }, itemStyle: { areaColor: "#fff", borderWidth: 0, borderColor: "#333" }, emphasis: { itemStyle: { areaColor: "#639fc0" }, label: { show: !0 } } }
        }
    }, function(e, t, r) {
        function n(e, t) {
            var r = e.getBoxLayoutParams(),
                n = s.getLayoutRect(r, { width: t.getWidth(), height: t.getHeight() });
            n.y = t.getHeight() - n.y - n.height, this.viewGL.setViewport(n.x, n.y, n.width, n.height, t.getDevicePixelRatio());
            var i = this.getGeoBoundingRect(),
                a = i.width / i.height * (e.get("aspectScale") || .75),
                o = e.get("boxWidth"),
                u = e.get("boxDepth"),
                h = e.get("boxHeight");
            null == h && (h = 5), isNaN(o) && isNaN(u) && (o = 100), isNaN(u) ? u = o / a : isNaN(o) && (o = u / a), this.setSize(o, h, u), this.regionHeight = e.get("regionHeight"), this.altitudeAxis && this.altitudeAxis.setExtent(0, Math.max(h - this.regionHeight, 0))
        }

        function i(e, t) {
            var r = [1 / 0, -1 / 0];
            if (e.eachSeries(function(e) {
                    if (e.coordinateSystem === this && "series.map3D" !== e.type) {
                        var t = e.getData(),
                            n = e.coordDimToDataDim("alt")[0];
                        if (n) {
                            var i = t.getDataExtent(n, !0);
                            r[0] = Math.min(r[0], i[0]), r[1] = Math.max(r[1], i[1])
                        }
                    }
                }, this), r && isFinite(r[1] - r[0])) {
                var n = o.helper.createScale(r, { type: "value", min: "dataMin", max: "dataMax" });
                this.altitudeAxis = new o.Axis("altitude", n), this.resize(this.model, t)
            }
        }
        var a = r(63),
            o = r(0),
            s = r(42),
            u = r(21),
            h = r(4),
            l = 0,
            c = {
                dimensions: a.prototype.dimensions,
                create: function(e, t) {
                    function r(e, r) {
                        var o = c.createGeo3D(e);
                        e.__viewGL = e.__viewGL || new u, o.viewGL = e.__viewGL, e.coordinateSystem = o, o.model = e, a.push(o), o.resize = n, o.resize(e, t), o.update = i
                    }
                    var a = [];
                    if (!o.getMap) throw new Error("geo3D component depends on geo component");
                    return e.eachComponent("geo3D", function(e, t) { r(e, t) }), e.eachSeriesByType("map3D", function(e, t) {
                        var n = e.get("coordinateSystem");
                        null == n && (n = "geo3D"), "geo3D" === n && r(e, t)
                    }), e.eachSeries(function(t) {
                        if ("geo3D" === t.get("coordinateSystem")) {
                            if ("series.map3D" === t.type) return;
                            var r = t.getReferringComponents("geo3D")[0];
                            if (r || (r = e.getComponent("geo3D")), !r) throw new Error('geo "' + h.firstNotNull(t.get("geo3DIndex"), t.get("geo3DId"), 0) + '" not found');
                            t.coordinateSystem = r.coordinateSystem
                        }
                    }), a
                },
                createGeo3D: function(e) { var t, r = e.get("map"); return "string" == typeof r ? (t = r, r = o.getMap(r)) : r && r.features && (r = { geoJson: r }), null == t && (t = "GEO_ANONYMOUS_" + l++), new a(t + l++, t, r && r.geoJson, r && r.specialAreas, e.get("nameMap")) }
            };
        o.registerCoordinateSystem("geo3D", c), e.exports = c
    }, function(e, t) {
        function r(e, t, r) {
            var n = e[t];
            e[t] = e[r], e[r] = n
        }

        function n(e, t, n, i, a) {
            var o = n,
                s = e[t];
            r(e, t, i);
            for (var u = n; u < i; u++) a(e[u], s) < 0 && (r(e, u, o), o++);
            return r(e, i, o), o
        }

        function i(e, t, r, a) {
            if (r < a) {
                var o = Math.floor((r + a) / 2),
                    s = n(e, o, r, a, t);
                i(e, t, r, s - 1), i(e, t, s + 1, a)
            }
        }

        function a() { this._parts = [] }
        a.prototype.step = function(e, t, r) {
            var a = e.length;
            if (0 === r) {
                this._parts = [], this._sorted = !1;
                var o = Math.floor(a / 2);
                this._parts.push({ pivot: o, left: 0, right: a - 1 }), this._currentSortPartIdx = 0
            }
            if (!this._sorted) {
                var s = this._parts;
                if (0 === s.length) return this._sorted = !0, !0;
                if (s.length < 512) {
                    for (var u = 0; u < s.length; u++) s[u].pivot = n(e, s[u].pivot, s[u].left, s[u].right, t);
                    for (var h = [], u = 0; u < s.length; u++) {
                        var l = s[u].left,
                            c = s[u].pivot - 1;
                        c > l && h.push({ pivot: Math.floor((c + l) / 2), left: l, right: c });
                        var l = s[u].pivot + 1,
                            c = s[u].right;
                        c > l && h.push({ pivot: Math.floor((c + l) / 2), left: l, right: c })
                    }
                    s = this._parts = h
                } else
                    for (var u = 0; u < Math.floor(s.length / 10); u++) { var d = s.length - 1 - this._currentSortPartIdx; if (i(e, t, s[d].left, s[d].right), ++this._currentSortPartIdx === s.length) return this._sorted = !0, !0 }
                return !1
            }
        }, a.sort = i, e.exports = a
    }, function(e, t, r) {
        function n(e, t, r, n, i, a, o) { this._zr = e, this._x = 0, this._y = 0, this._rowHeight = 0, this.width = n, this.height = i, this.offsetX = t, this.offsetY = r, this.dpr = o, this.gap = a }

        function i(e) {
            e = e || {}, e.width = e.width || 512, e.height = e.height || 512, e.devicePixelRatio = e.devicePixelRatio || 1, e.gap = null == e.gap ? 2 : e.gap;
            var t = document.createElement("canvas");
            t.width = e.width * e.devicePixelRatio, t.height = e.height * e.devicePixelRatio, this._canvas = t, this._texture = new o({ image: t, flipY: !1 });
            var r = this;
            this._zr = a.zrender.init(t);
            var i = this._zr.refreshImmediately;
            this._zr.refreshImmediately = function() { i.call(this), r._texture.dirty(), r.onupdate && r.onupdate() }, this._dpr = e.devicePixelRatio, this._coords = {}, this.onupdate = e.onupdate, this._gap = e.gap, this._textureAtlasNodes = [new n(this._zr, 0, 0, e.width, e.height, this._gap, this._dpr)], this._nodeWidth = e.width, this._nodeHeight = e.height, this._currentNodeIdx = 0
        }
        var a = r(0),
            o = r(5);
        n.prototype = {
            constructor: n,
            clear: function() { this._x = 0, this._y = 0, this._rowHeight = 0 },
            add: function(e, t, r) {
                var n = e.getBoundingRect();
                null == t && (t = n.width), null == r && (r = n.height), t *= this.dpr, r *= this.dpr, this._fitElement(e, t, r);
                var i = this._x,
                    a = this._y,
                    o = this.width * this.dpr,
                    s = this.height * this.dpr,
                    u = this.gap;
                if (i + t + u > o && (i = this._x = 0, a += this._rowHeight + u, this._y = a, this._rowHeight = 0), this._x += t + u, this._rowHeight = Math.max(this._rowHeight, r), a + r + u > s) return null;
                e.position[0] += this.offsetX * this.dpr + i, e.position[1] += this.offsetY * this.dpr + a, this._zr.add(e);
                var h = [this.offsetX / this.width, this.offsetY / this.height];
                return [
                    [i / o + h[0], a / s + h[1]],
                    [(i + t) / o + h[0], (a + r) / s + h[1]]
                ]
            },
            _fitElement: function(e, t, r) {
                var n = e.getBoundingRect(),
                    i = t / n.width,
                    a = r / n.height;
                e.position = [-n.x * i, -n.y * a], e.scale = [i, a], e.update()
            }
        }, i.prototype = {
            clear: function() {
                for (var e = 0; e < this._textureAtlasNodes.length; e++) this._textureAtlasNodes[e].clear();
                this._currentNodeIdx = 0, this._zr.clear(), this._coords = {}
            },
            getWidth: function() { return this._width },
            getHeight: function() { return this._height },
            getTexture: function() { return this._texture },
            getDevicePixelRatio: function() { return this._dpr },
            getZr: function() { return this._zr },
            _getCurrentNode: function() { return this._textureAtlasNodes[this._currentNodeIdx] },
            _expand: function() {
                if (this._currentNodeIdx++, this._textureAtlasNodes[this._currentNodeIdx]) return this._textureAtlasNodes[this._currentNodeIdx];
                var e = 4096 / this._dpr,
                    t = this._textureAtlasNodes,
                    r = t.length,
                    i = r * this._nodeWidth % e,
                    a = Math.floor(r * this._nodeWidth / e) * this._nodeHeight;
                if (!(a >= e)) {
                    var o = (i + this._nodeWidth) * this._dpr,
                        s = (a + this._nodeHeight) * this._dpr;
                    try { this._zr.resize({ width: o, height: s }) } catch (e) { this._canvas.width = o, this._canvas.height = s }
                    var u = new n(this._zr, i, a, this._nodeWidth, this._nodeHeight, this._gap, this._dpr);
                    return this._textureAtlasNodes.push(u), u
                }
            },
            add: function(e, t, r) {
                if (this._coords[e.id]) return this._coords[e.id];
                var n = this._getCurrentNode().add(e, t, r);
                if (!n) {
                    var i = this._expand();
                    if (!i) return;
                    n = i.add(e, t, r)
                }
                return this._coords[e.id] = n, n
            },
            getCoordsScale: function() { var e = this._dpr; return [this._nodeWidth / this._canvas.width * e, this._nodeHeight / this._canvas.height * e] },
            getCoords: function(e) { return this._coords[e] }
        }, e.exports = i
    }, function(e, t, r) {
        function n(e) { return e.replace(/^\s+/, "").replace(/\s+$/, "") }

        function i(e) { return Math.floor(Math.log(e) / Math.LN10) }
        var a = r(15),
            o = {};
        o.linearMap = function(e, t, r, n) {
            var i = t[1] - t[0],
                a = r[1] - r[0];
            if (0 === i) return 0 === a ? r[0] : (r[0] + r[1]) / 2;
            if (n)
                if (i > 0) { if (e <= t[0]) return r[0]; if (e >= t[1]) return r[1] } else { if (e >= t[0]) return r[0]; if (e <= t[1]) return r[1] }
            else { if (e === t[0]) return r[0]; if (e === t[1]) return r[1] }
            return (e - t[0]) / i * a + r[0]
        }, o.parsePercent = function(e, t) {
            switch (e) {
                case "center":
                case "middle":
                    e = "50%";
                    break;
                case "left":
                case "top":
                    e = "0%";
                    break;
                case "right":
                case "bottom":
                    e = "100%"
            }
            return "string" == typeof e ? n(e).match(/%$/) ? parseFloat(e) / 100 * t : parseFloat(e) : null == e ? NaN : +e
        }, o.round = function(e, t, r) { return null == t && (t = 10), t = Math.min(Math.max(0, t), 20), e = (+e).toFixed(t), r ? e : +e }, o.asc = function(e) { return e.sort(function(e, t) { return e - t }), e }, o.getPrecision = function(e) { if (e = +e, isNaN(e)) return 0; for (var t = 1, r = 0; Math.round(e * t) / t !== e;) t *= 10, r++; return r }, o.getPrecisionSafe = function(e) {
            var t = e.toString(),
                r = t.indexOf("e");
            if (r > 0) { var n = +t.slice(r + 1); return n < 0 ? -n : 0 }
            var i = t.indexOf(".");
            return i < 0 ? 0 : t.length - 1 - i
        }, o.getPixelPrecision = function(e, t) {
            var r = Math.log,
                n = Math.LN10,
                i = Math.floor(r(e[1] - e[0]) / n),
                a = Math.round(r(Math.abs(t[1] - t[0])) / n),
                o = Math.min(Math.max(-i + a, 0), 20);
            return isFinite(o) ? o : 20
        }, o.getPercentWithPrecision = function(e, t, r) { if (!e[t]) return 0; var n = a.reduce(e, function(e, t) { return e + (isNaN(t) ? 0 : t) }, 0); if (0 === n) return 0; for (var i = Math.pow(10, r), o = a.map(e, function(e) { return (isNaN(e) ? 0 : e) / n * i * 100 }), s = 100 * i, u = a.map(o, function(e) { return Math.floor(e) }), h = a.reduce(u, function(e, t) { return e + t }, 0), l = a.map(o, function(e, t) { return e - u[t] }); h < s;) { for (var c = Number.NEGATIVE_INFINITY, d = null, f = 0, p = l.length; f < p; ++f) l[f] > c && (c = l[f], d = f);++u[d], l[d] = 0, ++h } return u[t] / i }, o.MAX_SAFE_INTEGER = 9007199254740991, o.remRadian = function(e) { var t = 2 * Math.PI; return (e % t + t) % t }, o.isRadianAroundZero = function(e) { return e > -1e-4 && e < 1e-4 };
        var s = /^(?:(\d{4})(?:[-\/](\d{1,2})(?:[-\/](\d{1,2})(?:[T ](\d{1,2})(?::(\d\d)(?::(\d\d)(?:[.,](\d+))?)?)?(Z|[\+\-]\d\d:?\d\d)?)?)?)?)?$/;
        o.getTimezoneOffset = function() { return (new Date).getTimezoneOffset() }, o.parseDate = function(e) {
            if (e instanceof Date) return e;
            if ("string" == typeof e) {
                var t = s.exec(e);
                if (!t) return new Date(NaN);
                var r = o.getTimezoneOffset(),
                    n = t[8] ? "Z" === t[8].toUpperCase() ? r : 60 * +t[8].slice(0, 3) + r : 0;
                return new Date(+t[1], +(t[2] || 1) - 1, +t[3] || 1, +t[4] || 0, +(t[5] || 0) - n, +t[6] || 0, +t[7] || 0)
            }
            return null == e ? new Date(NaN) : new Date(Math.round(e))
        }, o.quantity = function(e) { return Math.pow(10, i(e)) }, o.nice = function(e, t) {
            var r, n = i(e),
                a = Math.pow(10, n),
                o = e / a;
            return r = t ? o < 1.5 ? 1 : o < 2.5 ? 2 : o < 4 ? 3 : o < 7 ? 5 : 10 : o < 1 ? 1 : o < 2 ? 2 : o < 3 ? 3 : o < 5 ? 5 : 10, e = r * a, n >= -20 ? +e.toFixed(n < 0 ? -n : 0) : e
        }, o.reformIntervals = function(e) {
            function t(e, r, n) { return e.interval[n] < r.interval[n] || e.interval[n] === r.interval[n] && (e.close[n] - r.close[n] == (n ? -1 : 1) || !n && t(e, r, 1)) }
            e.sort(function(e, r) { return t(e, r, 0) ? -1 : 1 });
            for (var r = -1 / 0, n = 1, i = 0; i < e.length;) {
                for (var a = e[i].interval, o = e[i].close, s = 0; s < 2; s++) a[s] <= r && (a[s] = r, o[s] = s ? 1 : 1 - n), r = a[s], n = o[s];
                a[0] === a[1] && o[0] * o[1] != 1 ? e.splice(i, 1) : i++
            }
            return e
        }, o.isNumeric = function(e) { return e - parseFloat(e) >= 0 }, e.exports = o
    }, function(e, t, r) {
        "use strict";
        var n = r(35),
            i = r(9),
            a = r(54),
            o = r(56),
            s = r(1),
            u = s.vec3,
            h = s.vec4,
            l = n.extend(function() { return { projectionMatrix: new i, invProjectionMatrix: new i, viewMatrix: new i, frustum: new a } }, function() { this.update(!0) }, {
                update: function(e) { n.prototype.update.call(this, e), i.invert(this.viewMatrix, this.worldTransform), this.updateProjectionMatrix(), i.invert(this.invProjectionMatrix, this.projectionMatrix), this.frustum.setFromProjection(this.projectionMatrix) },
                setViewMatrix: function(e) { i.copy(this.viewMatrix, e), i.invert(this.worldTransform, e), this.decomposeWorldTransform() },
                decomposeProjectionMatrix: function() {},
                setProjectionMatrix: function(e) { i.copy(this.projectionMatrix, e), i.invert(this.invProjectionMatrix, e), this.decomposeProjectionMatrix() },
                updateProjectionMatrix: function() {},
                castRay: function() {
                    var e = h.create();
                    return function(t, r) {
                        var n = void 0 !== r ? r : new o,
                            i = t._array[0],
                            a = t._array[1];
                        return h.set(e, i, a, -1, 1), h.transformMat4(e, e, this.invProjectionMatrix._array), h.transformMat4(e, e, this.worldTransform._array), u.scale(n.origin._array, e, 1 / e[3]), h.set(e, i, a, 1, 1), h.transformMat4(e, e, this.invProjectionMatrix._array), h.transformMat4(e, e, this.worldTransform._array), u.scale(e, e, 1 / e[3]), u.sub(n.direction._array, e, n.origin._array), u.normalize(n.direction._array, n.direction._array), n.direction._dirty = !0, n.origin._dirty = !0, n
                    }
                }()
            });
        e.exports = l
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r) { this.availableAttributes = e, this.availableAttributeSymbols = t, this.indicesBuffer = r, this.vao = null }
        var i, a = r(35),
            o = r(11),
            s = r(17),
            u = 0,
            h = null,
            l = !0,
            c = function() { this.triangleCount = 0, this.vertexCount = 0, this.drawCallCount = 0 },
            d = a.extend({ material: null, geometry: null, mode: o.TRIANGLES, _drawCache: null, _renderInfo: null }, function() { this._drawCache = {}, this._renderInfo = new c }, {
                renderOrder: 0,
                lineWidth: 1,
                culling: !0,
                cullFace: o.BACK,
                frontFace: o.CCW,
                frustumCulling: !0,
                receiveShadow: !0,
                castShadow: !0,
                ignorePicking: !1,
                ignorePreZ: !1,
                isRenderable: function() { return this.geometry && this.material && !this.invisible && this.geometry.vertexCount > 0 },
                beforeRender: function(e) {},
                afterRender: function(e, t) {},
                getBoundingBox: function(e, t) { return t = a.prototype.getBoundingBox.call(this, e, t), this.geometry && this.geometry.boundingBox && t.union(this.geometry.boundingBox), t },
                render: function(e, t) {
                    var t = t || this.material.shader,
                        r = this.geometry,
                        a = this.mode,
                        c = r.vertexCount,
                        d = r.isUseIndices(),
                        f = s.getExtension(e, "OES_element_index_uint"),
                        p = f && c > 65535,
                        _ = p ? e.UNSIGNED_INT : e.UNSIGNED_SHORT,
                        m = s.getExtension(e, "OES_vertex_array_object"),
                        g = !r.dynamic,
                        v = this._renderInfo;
                    v.vertexCount = c, v.triangleCount = 0, v.drawCallCount = 0;
                    var y = !1;
                    if (i = e.__GLID__ + "-" + r.__GUID__ + "-" + t.__GUID__, i !== u ? y = !0 : (c > 65535 && !f && d || m && g || r._cache.isDirty()) && (y = !0), u = i, y) {
                        var x = this._drawCache[i];
                        if (!x) {
                            var T = r.getBufferChunks(e);
                            if (!T) return;
                            x = [];
                            for (var b = 0; b < T.length; b++) {
                                for (var w = T[b], E = w.attributeBuffers, S = w.indicesBuffer, A = [], M = [], N = 0; N < E.length; N++) {
                                    var C, L = E[N],
                                        D = L.name,
                                        I = L.semantic;
                                    if (I) {
                                        var R = t.attribSemantics[I];
                                        C = R && R.symbol
                                    } else C = D;
                                    C && t.attributeTemplates[C] && (A.push(L), M.push(C))
                                }
                                var P = new n(A, M, S);
                                x.push(P)
                            }
                            g && (this._drawCache[i] = x)
                        }
                        for (var O = 0; O < x.length; O++) {
                            var P = x[O],
                                F = !0;
                            m && g && (null == P.vao ? P.vao = m.createVertexArrayOES() : F = !1, m.bindVertexArrayOES(P.vao));
                            var A = P.availableAttributes,
                                S = P.indicesBuffer;
                            if (F)
                                for (var B = t.enableAttributes(e, P.availableAttributeSymbols, m && g && P.vao), N = 0; N < A.length; N++) {
                                    var U = B[N];
                                    if (-1 !== U) {
                                        var z, L = A[N],
                                            G = L.buffer,
                                            k = L.size;
                                        switch (L.type) {
                                            case "float":
                                                z = e.FLOAT;
                                                break;
                                            case "byte":
                                                z = e.BYTE;
                                                break;
                                            case "ubyte":
                                                z = e.UNSIGNED_BYTE;
                                                break;
                                            case "short":
                                                z = e.SHORT;
                                                break;
                                            case "ushort":
                                                z = e.UNSIGNED_SHORT;
                                                break;
                                            default:
                                                z = e.FLOAT
                                        }
                                        e.bindBuffer(e.ARRAY_BUFFER, G), e.vertexAttribPointer(U, k, z, !1, 0, 0)
                                    }
                                }
                            a != o.LINES && a != o.LINE_STRIP && a != o.LINE_LOOP || e.lineWidth(this.lineWidth), h = S, l = r.isUseIndices(), l ? (F && e.bindBuffer(e.ELEMENT_ARRAY_BUFFER, S.buffer), e.drawElements(a, S.count, _, 0), v.triangleCount += S.count / 3) : e.drawArrays(a, 0, c), m && g && m.bindVertexArrayOES(null), v.drawCallCount++
                        }
                    } else l ? (e.drawElements(a, h.count, _, 0), v.triangleCount = h.count / 3) : e.drawArrays(a, 0, c), v.drawCallCount = 1;
                    return v
                },
                clone: function() {
                    var e = ["castShadow", "receiveShadow", "mode", "culling", "cullFace", "frontFace", "frustumCulling"];
                    return function() {
                        var t = a.prototype.clone.call(this);
                        t.geometry = this.geometry, t.material = this.material;
                        for (var r = 0; r < e.length; r++) {
                            var n = e[r];
                            t[n] !== this[n] && (t[n] = this[n])
                        }
                        return t
                    }
                }()
            });
        d.beforeFrame = function() { u = 0 }, d.POINTS = o.POINTS, d.LINES = o.LINES, d.LINE_LOOP = o.LINE_LOOP, d.LINE_STRIP = o.LINE_STRIP, d.TRIANGLES = o.TRIANGLES, d.TRIANGLE_STRIP = o.TRIANGLE_STRIP, d.TRIANGLE_FAN = o.TRIANGLE_FAN, d.BACK = o.BACK, d.FRONT = o.FRONT, d.FRONT_AND_BACK = o.FRONT_AND_BACK, d.CW = o.CW, d.CCW = o.CCW, d.RenderInfo = c, e.exports = d
    }, function(e, t, r) {
        "use strict";
        var n = r(198),
            i = r(72),
            a = r(10),
            o = n.extend(function() { return { _outputs: [], _texturePool: new i, _frameBuffer: new a({ depthBuffer: !1 }) } }, { addNode: function(e) { n.prototype.addNode.call(this, e), e._compositor = this }, render: function(e, t) { if (this._dirty) { this.update(), this._dirty = !1, this._outputs.length = 0; for (var r = 0; r < this.nodes.length; r++) this.nodes[r].outputs || this._outputs.push(this.nodes[r]) } for (var r = 0; r < this.nodes.length; r++) this.nodes[r].beforeFrame(); for (var r = 0; r < this._outputs.length; r++) this._outputs[r].updateReference(); for (var r = 0; r < this._outputs.length; r++) this._outputs[r].render(e, t); for (var r = 0; r < this.nodes.length; r++) this.nodes[r].afterFrame() }, allocateTexture: function(e) { return this._texturePool.get(e) }, releaseTexture: function(e) { this._texturePool.put(e) }, getFrameBuffer: function() { return this._frameBuffer }, dispose: function(e) { this._texturePool.clear(e.gl || e) } });
        e.exports = o
    }, function(e, t, r) {
        "use strict";

        function n(e) { u.defaultsWithPropList(e, l, c), i(e); for (var t = "", r = 0; r < c.length; r++) { t += e[c[r]].toString() } return t }

        function i(e) {
            var t = a(e.width, e.height);
            e.format === s.DEPTH_COMPONENT && (e.useMipmap = !1), t && e.useMipmap || (e.minFilter == s.NEAREST_MIPMAP_NEAREST || e.minFilter == s.NEAREST_MIPMAP_LINEAR ? e.minFilter = s.NEAREST : e.minFilter != s.LINEAR_MIPMAP_LINEAR && e.minFilter != s.LINEAR_MIPMAP_NEAREST || (e.minFilter = s.LINEAR), e.wrapS = s.CLAMP_TO_EDGE, e.wrapT = s.CLAMP_TO_EDGE)
        }

        function a(e, t) { return 0 == (e & e - 1) && 0 == (t & t - 1) }
        var o = r(5),
            s = r(11),
            u = r(27),
            h = function() { this._pool = {}, this._allocatedTextures = [] };
        h.prototype = {
            constructor: h,
            get: function(e) {
                var t = n(e);
                this._pool.hasOwnProperty(t) || (this._pool[t] = []);
                var r = this._pool[t];
                if (!r.length) { var i = new o(e); return this._allocatedTextures.push(i), i }
                return r.pop()
            },
            put: function(e) {
                var t = n(e);
                this._pool.hasOwnProperty(t) || (this._pool[t] = []), this._pool[t].push(e)
            },
            clear: function(e) {
                for (var t = 0; t < this._allocatedTextures.length; t++) this._allocatedTextures[t].dispose(e);
                this._pool = {}, this._allocatedTextures = []
            }
        };
        var l = { width: 512, height: 512, type: s.UNSIGNED_BYTE, format: s.RGBA, wrapS: s.CLAMP_TO_EDGE, wrapT: s.CLAMP_TO_EDGE, minFilter: s.LINEAR_MIPMAP_LINEAR, magFilter: s.LINEAR, useMipmap: !0, anisotropic: 1, flipY: !0, unpackAlignment: 4, premultiplyAlpha: !1 },
            c = Object.keys(l);
        e.exports = h
    }, function(e, t, r) {
        "use strict";

        function n(e) {
            var t = new XMLHttpRequest;
            t.open("get", e.url), t.responseType = e.responseType || "text", e.onprogress && (t.onprogress = function(t) {
                if (t.lengthComputable) {
                    var r = t.loaded / t.total;
                    e.onprogress(r, t.loaded, t.total)
                } else e.onprogress(null)
            }), t.onload = function(r) { e.onload && e.onload(t.response) }, e.onerror && (t.onerror = e.onerror), t.send(null)
        }
        e.exports = { get: n }
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r) {
            l.identity();
            var n = new a({ widthSegments: t, heightSegments: r });
            switch (e) {
                case "px":
                    o.translate(l, l, s.POSITIVE_X), o.rotateY(l, l, Math.PI / 2);
                    break;
                case "nx":
                    o.translate(l, l, s.NEGATIVE_X), o.rotateY(l, l, -Math.PI / 2);
                    break;
                case "py":
                    o.translate(l, l, s.POSITIVE_Y), o.rotateX(l, l, -Math.PI / 2);
                    break;
                case "ny":
                    o.translate(l, l, s.NEGATIVE_Y), o.rotateX(l, l, Math.PI / 2);
                    break;
                case "pz":
                    o.translate(l, l, s.POSITIVE_Z);
                    break;
                case "nz":
                    o.translate(l, l, s.NEGATIVE_Z), o.rotateY(l, l, Math.PI)
            }
            return n.applyTransform(l), n
        }
        var i = r(13),
            a = r(46),
            o = r(9),
            s = r(3),
            u = r(14),
            h = r(20),
            l = new o,
            c = i.extend({ widthSegments: 1, heightSegments: 1, depthSegments: 1, inside: !1 }, function() { this.build() }, {
                build: function() {
                    var e = { px: n("px", this.depthSegments, this.heightSegments), nx: n("nx", this.depthSegments, this.heightSegments), py: n("py", this.widthSegments, this.depthSegments), ny: n("ny", this.widthSegments, this.depthSegments), pz: n("pz", this.widthSegments, this.heightSegments), nz: n("nz", this.widthSegments, this.heightSegments) },
                        t = ["position", "texcoord0", "normal"],
                        r = 0,
                        i = 0;
                    for (var a in e) r += e[a].vertexCount, i += e[a].indices.length;
                    for (var o = 0; o < t.length; o++) this.attributes[t[o]].init(r);
                    this.indices = new h.Uint16Array(i);
                    var s = 0,
                        l = 0;
                    for (var a in e) {
                        for (var c = e[a], o = 0; o < t.length; o++)
                            for (var d = t[o], f = c.attributes[d].value, p = c.attributes[d].size, _ = "normal" === d, m = 0; m < f.length; m++) {
                                var g = f[m];
                                this.inside && _ && (g = -g), this.attributes[d].value[m + p * l] = g
                            }
                        for (var m = 0; m < c.indices.length; m++) this.indices[m + s] = l + c.indices[m];
                        s += c.indices.length, l += c.vertexCount
                    }
                    this.boundingBox = new u, this.boundingBox.max.set(1, 1, 1), this.boundingBox.min.set(-1, -1, -1)
                }
            });
        e.exports = c
    }, function(e, t, r) {
        "use strict";
        var n = r(13),
            i = r(1),
            a = (i.vec3, i.vec2, r(14)),
            o = n.extend({ widthSegments: 20, heightSegments: 20, phiStart: 0, phiLength: 2 * Math.PI, thetaStart: 0, thetaLength: Math.PI, radius: 1 }, function() { this.build() }, {
                build: function() {
                    var e = this.heightSegments,
                        t = this.widthSegments,
                        r = this.attributes.position,
                        n = this.attributes.texcoord0,
                        i = this.attributes.normal,
                        o = (t + 1) * (e + 1);
                    r.init(o), n.init(o), i.init(o);
                    var s, u, h, l, c, d, f, p = o > 65535 ? Uint32Array : Uint16Array,
                        _ = this.indices = new p(t * e * 6),
                        m = this.radius,
                        g = this.phiStart,
                        v = this.phiLength,
                        y = this.thetaStart,
                        x = this.thetaLength,
                        m = this.radius,
                        T = [],
                        b = [],
                        w = 0,
                        E = 1 / m;
                    for (f = 0; f <= e; f++)
                        for (d = 0; d <= t; d++) l = d / t, c = f / e, s = -m * Math.cos(g + l * v) * Math.sin(y + c * x), u = m * Math.cos(y + c * x), h = m * Math.sin(g + l * v) * Math.sin(y + c * x), T[0] = s, T[1] = u, T[2] = h, b[0] = l, b[1] = c, r.set(w, T), n.set(w, b), T[0] *= E, T[1] *= E, T[2] *= E, i.set(w, T), w++;
                    var S, A, M, N, C = t + 1,
                        L = 0;
                    for (f = 0; f < e; f++)
                        for (d = 0; d < t; d++) A = f * C + d, S = f * C + d + 1, N = (f + 1) * C + d + 1, M = (f + 1) * C + d, _[L++] = S, _[L++] = A, _[L++] = N, _[L++] = A, _[L++] = M, _[L++] = N;
                    this.boundingBox = new a, this.boundingBox.max.set(m, m, m), this.boundingBox.min.set(-m, -m, -m)
                }
            });
        e.exports = o
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = r(3),
            a = n.extend({ shadowBias: .001, shadowSlopeScale: 2, shadowCascade: 1, cascadeSplitLogFactor: .2 }, {
                type: "DIRECTIONAL_LIGHT",
                uniformTemplates: {
                    directionalLightDirection: { type: "3f", value: function(e) { return e.__dir = e.__dir || new i, e.__dir.copy(e.worldTransform.z).normalize().negate()._array } },
                    directionalLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    }
                },
                clone: function() { var e = n.prototype.clone.call(this); return e.shadowBias = this.shadowBias, e.shadowSlopeScale = this.shadowSlopeScale, e }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = n.extend({ range: 100, castShadow: !1 }, {
                type: "POINT_LIGHT",
                uniformTemplates: {
                    pointLightPosition: { type: "3f", value: function(e) { return e.getWorldPosition()._array } },
                    pointLightRange: { type: "1f", value: function(e) { return e.range } },
                    pointLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    }
                },
                clone: function() { var e = n.prototype.clone.call(this); return e.range = this.range, e }
            });
        e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = r(3),
            a = n.extend({ range: 20, umbraAngle: 30, penumbraAngle: 45, falloffFactor: 2, shadowBias: 2e-4, shadowSlopeScale: 2 }, {
                type: "SPOT_LIGHT",
                uniformTemplates: {
                    spotLightPosition: { type: "3f", value: function(e) { return e.getWorldPosition()._array } },
                    spotLightRange: { type: "1f", value: function(e) { return e.range } },
                    spotLightUmbraAngleCosine: { type: "1f", value: function(e) { return Math.cos(e.umbraAngle * Math.PI / 180) } },
                    spotLightPenumbraAngleCosine: { type: "1f", value: function(e) { return Math.cos(e.penumbraAngle * Math.PI / 180) } },
                    spotLightFalloffFactor: { type: "1f", value: function(e) { return e.falloffFactor } },
                    spotLightDirection: { type: "3f", value: function(e) { return e.__dir = e.__dir || new i, e.__dir.copy(e.worldTransform.z).negate()._array } },
                    spotLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    }
                },
                clone: function() { var e = n.prototype.clone.call(this); return e.range = this.range, e.umbraAngle = this.umbraAngle, e.penumbraAngle = this.penumbraAngle, e.falloffFactor = this.falloffFactor, e.shadowBias = this.shadowBias, e.shadowSlopeScale = this.shadowSlopeScale, e }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(3),
            i = r(1),
            a = i.vec3,
            o = i.mat4,
            s = i.vec4,
            u = function(e, t) { this.normal = e || new n(0, 1, 0), this.distance = t || 0 };
        u.prototype = {
            constructor: u,
            distanceToPoint: function(e) { return a.dot(e._array, this.normal._array) - this.distance },
            projectPoint: function(e, t) { t || (t = new n); var r = this.distanceToPoint(e); return a.scaleAndAdd(t._array, e._array, this.normal._array, -r), t._dirty = !0, t },
            normalize: function() {
                var e = 1 / a.len(this.normal._array);
                a.scale(this.normal._array, e), this.distance *= e
            },
            intersectFrustum: function(e) {
                for (var t = e.vertices, r = this.normal._array, n = a.dot(t[0]._array, r) > this.distance, i = 1; i < 8; i++)
                    if (a.dot(t[i]._array, r) > this.distance != n) return !0
            },
            intersectLine: function() {
                var e = a.create();
                return function(t, r, i) {
                    var o = this.distanceToPoint(t),
                        s = this.distanceToPoint(r);
                    if (o > 0 && s > 0 || o < 0 && s < 0) return null;
                    var u = this.normal._array,
                        h = this.distance,
                        l = t._array;
                    a.sub(e, r._array, t._array), a.normalize(e, e);
                    var c = a.dot(u, e);
                    if (0 === c) return null;
                    i || (i = new n);
                    var d = (a.dot(u, l) - h) / c;
                    return a.scaleAndAdd(i._array, l, e, -d), i._dirty = !0, i
                }
            }(),
            applyTransform: function() {
                var e = o.create(),
                    t = s.create(),
                    r = s.create();
                return r[3] = 1,
                    function(n) { n = n._array, a.scale(r, this.normal._array, this.distance), s.transformMat4(r, r, n), this.distance = a.dot(r, this.normal._array), o.invert(e, n), o.transpose(e, e), t[3] = 0, a.copy(t, this.normal._array), s.transformMat4(t, t, e), a.copy(this.normal._array, t) }
            }(),
            copy: function(e) { a.copy(this.normal._array, e.normal._array), this.normal._dirty = !0, this.distance = e.distance },
            clone: function() { var e = new u; return e.copy(this), e }
        }, e.exports = u
    }, function(e, t) {
        var r = {};
        r.isPowerOfTwo = function(e) { return 0 == (e & e - 1) }, r.nextPowerOfTwo = function(e) { return e--, e |= e >> 1, e |= e >> 2, e |= e >> 4, e |= e >> 8, e |= e >> 16, ++e }, r.nearestPowerOfTwo = function(e) { return Math.pow(2, Math.round(Math.log(e) / Math.LN2)) }, e.exports = r
    }, function(e, t, r) {
        function n() { this._pool = {} }

        function i(e, t, r) { o[e] = { vertex: t, fragment: r } }
        var a = r(7),
            o = (r(27), {});
        n.prototype.get = function(e, t) {
            var r = [],
                n = {},
                i = {};
            "string" == typeof t ? r = Array.prototype.slice.call(arguments, 1) : "[object Object]" == Object.prototype.toString.call(t) ? (r = t.textures || [], n = t.vertexDefines || {}, i = t.fragmentDefines || {}) : t instanceof Array && (r = t);
            var s = Object.keys(n),
                u = Object.keys(i);
            r.sort(), s.sort(), u.sort();
            var h = [e];
            h = h.concat(r);
            for (var l = 0; l < s.length; l++) h.push(s[l], n[s[l]]);
            for (var l = 0; l < u.length; l++) h.push(u[l], i[u[l]]);
            var c = h.join("_");
            if (this._pool[c]) return this._pool[c];
            var d = o[e];
            if (!d) return void console.error('Shader "' + e + '" is not in the library');
            for (var f = new a({ vertex: d.vertex, fragment: d.fragment }), l = 0; l < r.length; l++) f.enableTexture(r[l]);
            for (var e in n) f.define("vertex", e, n[e]);
            for (var e in i) f.define("fragment", e, i[e]);
            return this._pool[c] = f, f
        }, n.prototype.clear = function() { this._pool = {} };
        var s = new n;
        e.exports = { createLibrary: function() { return new n }, get: function() { return s.get.apply(s, arguments) }, template: i, clear: function() { return s.clear() } }
    }, function(e, t) { e.exports = "@export qtek.prez.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position : POSITION;\n\n@import qtek.chunk.skinning_header\n\nvoid main()\n{\n\n vec3 skinnedPosition = position;\n\n#ifdef SKINNING\n\n @import qtek.chunk.skin_matrix\n\n skinnedPosition = (skinMatrixWS * vec4(position, 1.0)).xyz;\n#endif\n\n gl_Position = worldViewProjection * vec4(skinnedPosition, 1.0);\n}\n\n@end\n\n\n@export qtek.prez.fragment\n\nvoid main()\n{\n gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);\n}\n\n@end" }, function(e, t) { e.exports = "undefined" != typeof window && (window.requestAnimationFrame && window.requestAnimationFrame.bind(window) || window.msRequestAnimationFrame && window.msRequestAnimationFrame.bind(window) || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame) || function(e) { setTimeout(e, 16) } }, function(e, t, r) {
        "use strict";

        function n(e, t, r, n) { r < 0 && (e += r, r = -r), n < 0 && (t += n, n = -n), this.x = e, this.y = t, this.width = r, this.height = n }
        var i = r(242),
            a = r(241),
            o = i.applyTransform,
            s = Math.min,
            u = Math.max;
        n.prototype = {
            constructor: n,
            union: function(e) {
                var t = s(e.x, this.x),
                    r = s(e.y, this.y);
                this.width = u(e.x + e.width, this.x + this.width) - t, this.height = u(e.y + e.height, this.y + this.height) - r, this.x = t, this.y = r
            },
            applyTransform: function() {
                var e = [],
                    t = [],
                    r = [],
                    n = [];
                return function(i) {
                    if (i) {
                        e[0] = r[0] = this.x, e[1] = n[1] = this.y, t[0] = n[0] = this.x + this.width, t[1] = r[1] = this.y + this.height, o(e, e, i), o(t, t, i), o(r, r, i), o(n, n, i), this.x = s(e[0], t[0], r[0], n[0]), this.y = s(e[1], t[1], r[1], n[1]);
                        var a = u(e[0], t[0], r[0], n[0]),
                            h = u(e[1], t[1], r[1], n[1]);
                        this.width = a - this.x, this.height = h - this.y
                    }
                }
            }(),
            calculateTransform: function(e) {
                var t = this,
                    r = e.width / t.width,
                    n = e.height / t.height,
                    i = a.create();
                return a.translate(i, i, [-t.x, -t.y]), a.scale(i, i, [r, n]), a.translate(i, i, [e.x, e.y]), i
            },
            intersect: function(e) {
                if (!e) return !1;
                e instanceof n || (e = n.create(e));
                var t = this,
                    r = t.x,
                    i = t.x + t.width,
                    a = t.y,
                    o = t.y + t.height,
                    s = e.x,
                    u = e.x + e.width,
                    h = e.y,
                    l = e.y + e.height;
                return !(i < s || u < r || o < h || l < a)
            },
            contain: function(e, t) { var r = this; return e >= r.x && e <= r.x + r.width && t >= r.y && t <= r.y + r.height },
            clone: function() { return new n(this.x, this.y, this.width, this.height) },
            copy: function(e) { this.x = e.x, this.y = e.y, this.width = e.width, this.height = e.height },
            plain: function() { return { x: this.x, y: this.y, width: this.width, height: this.height } }
        }, n.create = function(e) { return new n(e.x, e.y, e.width, e.height) }, e.exports = n
    }, function(e, t, r) {
        var n = r(0);
        r(102), r(101), r(100), n.registerVisual(n.util.curry(r(18), "bar3D")), n.registerProcessor(function(e, t) {
            e.eachSeriesByType("bar3d", function(e) {
                var t = e.getData();
                t.filterSelf(function(e) { return t.hasValue(e) })
            })
        })
    }, function(e, t, r) {
        r(0);
        r(108), r(107)
    }, function(e, t, r) {
        function n() {}
        var i = r(0);
        r(113), r(114), i.registerVisual(i.util.curry(r(43), "graphGL", "circle", null)), i.registerVisual(i.util.curry(r(18), "graphGL")), i.registerVisual(function(e) {
            var t = {};
            e.eachSeriesByType("graphGL", function(e) {
                var r = e.getCategoriesData(),
                    n = e.getData(),
                    i = {};
                r.each(function(n) {
                    var a = r.getName(n);
                    i[a] = n;
                    var o = r.getItemModel(n),
                        s = o.get("itemStyle.color") || e.getColorFromPalette(a, t);
                    r.setItemVisual(n, "color", s)
                }), r.count() && n.each(function(e) {
                    var t = n.getItemModel(e),
                        a = t.getShallow("category");
                    null != a && ("string" == typeof a && (a = i[a]), n.getItemVisual(e, "color", !0) || n.setItemVisual(e, "color", r.getItemVisual(a, "color")))
                })
            })
        }), i.registerVisual(function(e) {
            e.eachSeriesByType("graphGL", function(e) {
                var t = e.getGraph(),
                    r = e.getEdgeData(),
                    n = "lineStyle.color".split("."),
                    i = "lineStyle.opacity".split(".");
                r.setVisual("color", e.get(n)), r.setVisual("opacity", e.get(i)), r.each(function(e) {
                    var a = r.getItemModel(e),
                        o = t.getEdgeByIndex(e),
                        s = a.get(n),
                        u = a.get(i);
                    switch (s) {
                        case "source":
                            s = o.node1.getVisual("color");
                            break;
                        case "target":
                            s = o.node2.getVisual("color")
                    }
                    o.setVisual("color", s), o.setVisual("opacity", u)
                })
            })
        }), i.registerAction({ type: "graphGLRoam", event: "graphglroam", update: "series.graphGL:roam" }, function(e, t) { t.eachComponent({ mainType: "series", query: e }, function(t) { t.setView(e) }) }), i.registerAction({ type: "graphGLStartLayout", event: "graphgllayoutstarted", update: "series.graphGL:startLayout" }, n), i.registerAction({ type: "graphGLStopLayout", event: "graphgllayoutstopped", update: "series.graphGL:stopLayout" }, n), i.registerAction({ type: "graphGLFocusNodeAdjacency", event: "graphGLFocusNodeAdjacency", update: "series.graphGL:focusNodeAdjacency" }, n), i.registerAction({ type: "graphGLUnfocusNodeAdjacency", event: "graphGLUnfocusNodeAdjacency", update: "series.graphGL:unfocusNodeAdjacency" }, n)
    }, function(e, t, r) {
        var n = r(0);
        r(118), r(119), n.registerVisual(n.util.curry(r(43), "line3D", "circle", null)), n.registerVisual(n.util.curry(r(18), "line3D")), n.registerLayout(function(e, t) {
            e.eachSeriesByType("line3D", function(e) {
                var t = e.getData(),
                    r = e.coordinateSystem;
                if (r) {
                    if ("cartesian3D" !== r.type) return;
                    var n = new Float32Array(3 * t.count()),
                        i = [],
                        a = [],
                        o = r.dimensions,
                        s = o.map(function(t) { return e.coordDimToDataDim(t)[0] });
                    r && t.each(s, function(e, t, o, s) { i[0] = e, i[1] = t, i[2] = o, r.dataToPoint(i, a), n[3 * s] = a[0], n[3 * s + 1] = a[1], n[3 * s + 2] = a[2] }), t.setLayout("points", n)
                }
            })
        })
    }, function(e, t, r) {
        var n = r(0);
        r(123), r(121), r(120), n.registerVisual(n.util.curry(r(18), "lines3D")), n.registerAction({ type: "lines3DPauseEffect", event: "lines3deffectpaused", update: "series.lines3D:pauseEffect" }, function() {}), n.registerAction({ type: "lines3DResumeEffect", event: "lines3deffectresumed", update: "series.lines3D:resumeEffect" }, function() {}), n.registerAction({ type: "lines3DToggleEffect", event: "lines3deffectchanged", update: "series.lines3D:toggleEffect" }, function() {})
    }, function(e, t, r) {
        function n(e, t) { for (var r = [], n = 0; n < e.length; n++) r.push(t.dataToPoint(e[n])); return r }

        function i(e, t) {
            for (var r = 0; r < e.regions.length; r++) {
                for (var i = e.regions[r], a = 0; a < i.geometries.length; a++) {
                    var o = i.geometries[a],
                        s = o.interiors;
                    if (o.exterior = n(o.exterior, t), s && s.length)
                        for (var u = 0; u < s.length; u++) o.interiors[u] = n(s[u], t)
                }
                i.center && (i.center = t.dataToPoint(i.center))
            }
        }
        var a = r(0);
        r(63);
        r(125), r(126);
        var o = r(65);
        a.registerVisual(a.util.curry(r(18), "map3D")), a.registerAction({ type: "map3DChangeCamera", event: "map3dcamerachanged", update: "series:updateCamera" }, function(e, t) { t.eachComponent({ mainType: "series", subType: "map3D", query: e }, function(t) { t.setView(e) }) }), a.registerLayout(function(e, t) {
            e.eachSeriesByType("map3D", function(e) {
                if ("mapbox" === e.get("coordinateSystem")) {
                    var t = o.createGeo3D(e);
                    t.extrudeY = !1, i(t, e.coordinateSystem), e.getData().setLayout("geo3D", t)
                }
            })
        })
    }, function(e, t, r) {
        var n = r(0);
        r(127), r(128), n.registerVisual(n.util.curry(r(43), "scatter3D", "circle", null)), n.registerVisual(n.util.curry(r(18), "scatter3D")), n.registerLayout(function(e, t) {
            e.eachSeriesByType("scatter3D", function(e) {
                var t = e.getData(),
                    r = e.coordinateSystem;
                if (r) {
                    var n = r.dimensions;
                    if (n.length < 3) return;
                    var i = n.map(function(t) { return e.coordDimToDataDim(t)[0] }),
                        a = new Float32Array(3 * t.count()),
                        o = [],
                        s = [];
                    r && t.each(i, function(e, t, n, i) { o[0] = e, o[1] = t, o[2] = n, r.dataToPoint(o, s), a[3 * i] = s[0], a[3 * i + 1] = s[1], a[3 * i + 2] = s[2] }), t.setLayout("points", a)
                }
            })
        })
    }, function(e, t, r) {
        var n = r(0);
        r(129), r(130), n.registerVisual(n.util.curry(r(43), "scatterGL", "circle", null)), n.registerVisual(n.util.curry(r(18), "scatterGL")), n.registerLayout(function(e, t) {
            e.eachSeriesByType("scatterGL", function(e) {
                var t = e.getData(),
                    r = e.coordinateSystem;
                if (r) {
                    var n = r.dimensions,
                        i = new Float32Array(2 * t.count());
                    if (1 === n.length) t.each(n[0], function(e, t) {
                        var n = r.dataToPoint(e);
                        i[2 * t] = n[0], i[2 * t + 1] = n[1]
                    });
                    else if (2 === n.length) {
                        var a = [];
                        t.each(n, function(e, t, n) {
                            a[0] = e, a[1] = t;
                            var o = r.dataToPoint(a);
                            i[2 * n] = o[0], i[2 * n + 1] = o[1]
                        })
                    }
                    t.setLayout("points", i)
                }
            })
        })
    }, function(e, t, r) {
        var n = r(0);
        r(131), r(132), r(133), n.registerVisual(n.util.curry(r(18), "surface"))
    }, function(e, t, r) {
        var n = r(0);
        r(134), r(135), r(65), n.registerAction({ type: "geo3DChangeCamera", event: "geo3dcamerachanged", update: "series:updateCamera" }, function(e, t) { t.eachComponent({ mainType: "geo3D", query: e }, function(t) { t.setView(e) }) })
    }, function(e, t, r) {
        var n = r(0);
        r(136), r(137), r(149), n.registerAction({ type: "globeChangeCamera", event: "globecamerachanged", update: "series:updateCamera" }, function(e, t) { t.eachComponent({ mainType: "globe", query: e }, function(t) { t.setView(e) }) }), n.registerAction({ type: "globeUpdateDisplacment", event: "globedisplacementupdated", update: "updateLayout" }, function(e, t) {})
    }, function(e, t, r) {
        r(138), r(141), r(142), r(152);
        var n = r(0);
        n.registerAction({ type: "grid3DChangeCamera", event: "grid3dcamerachanged", update: "series:updateCamera" }, function(e, t) { t.eachComponent({ mainType: "grid3D", query: e }, function(t) { t.setView(e) }) }), n.registerAction({ type: "grid3DShowAxisPointer", event: "grid3dshowaxispointer", update: "grid3D:showAxisPointer" }, function(e, t) {}), n.registerAction({ type: "grid3DHideAxisPointer", event: "grid3dhideaxispointer", update: "grid3D:hideAxisPointer" }, function(e, t) {})
    }, function(e, t, r) {
        var n = r(0);
        r(154), r(146), r(147), n.registerAction({ type: "mapboxChangeCamera", event: "mapboxcamerachanged", update: "mapbox:updateCamera" }, function(e, t) { t.eachComponent({ mainType: "mapbox", query: e }, function(t) { t.setMapboxCameraOption(e) }) })
    }, function(e, t, r) {
        function n(e) { throw new Error(e + " version is too old, needs " + l[e] + " or higher") }

        function i(e, t) { e.replace(".", "") - 0 < l[t].replace(".", "") - 0 && n(t), console.log("Loaded " + t + ", version " + e) }

        function a(e) { this._layers = {}, this._zr = e }
        var o = { version: "1.0.0-beta.5", dependencies: { echarts: "3.7.1", qtek: "0.4.3" } },
            s = r(0),
            u = r(235),
            h = r(155),
            l = o.dependencies;
        i(u, "qtek"), i(s.version, "echarts"), a.prototype.update = function(e, t) {
            function r(e) {
                var t = e.get("zlevel"),
                    r = i._layers,
                    n = r[t];
                if (!n) {
                    if (n = r[t] = new h("gl-" + t, a), a.painter.isSingleCanvas()) {
                        n.virtual = !0;
                        var o = new s.graphic.Image({ z: 1e4, style: { image: n.renderer.canvas }, silent: !0 });
                        n.__hostImage = o, a.add(o)
                    }
                    a.painter.insertLayer(t, n)
                }
                return n.__hostImage && n.__hostImage.setStyle({ width: n.renderer.getWidth(), height: n.renderer.getHeight() }), n
            }

            function n(e, t) { e && e.traverse(function(e) { e.isRenderable && e.isRenderable() && (e.ignorePicking = null != e.$ignorePicking ? e.$ignorePicking : t) }) }
            var i = this,
                a = t.getZr();
            if (!a.getWidth() || !a.getHeight()) return void console.warn("Dom has no width or height");
            for (var o in this._layers) this._layers[o].removeViewsAll();
            e.eachComponent(function(i, a) {
                if ("series" !== i) {
                    var o = t.getViewOfComponentModel(a),
                        s = a.coordinateSystem;
                    if (o.__ecgl__) {
                        var u;
                        if (s) {
                            if (!s.viewGL) return void console.error("Can't find viewGL in coordinateSystem of component " + a.id);
                            u = s.viewGL
                        } else {
                            if (!a.viewGL) return void console.error("Can't find viewGL of component " + a.id);
                            u = s.viewGL
                        }
                        var u = s.viewGL,
                            h = r(a);
                        h.addView(u), o.afterRender && o.afterRender(a, e, t, h), n(o.groupGL, a.get("silent"))
                    }
                }
            }), e.eachSeries(function(i) {
                var a = t.getViewOfSeriesModel(i),
                    o = i.coordinateSystem;
                if (a.__ecgl__) {
                    if (o && !o.viewGL && !a.viewGL) return void console.error("Can't find viewGL of series " + a.id);
                    var s = o && o.viewGL || a.viewGL,
                        u = r(i);
                    u.addView(s), a.afterRender && a.afterRender(i, e, t, u), n(a.groupGL, i.get("silent"))
                }
            })
        };
        var c = s.init;
        s.init = function() {
            var e = c.apply(this, arguments);
            return e.getZr().painter.getRenderedCanvas = function(e) {
                function t(e, t) {
                    var r = u._zlevelList;
                    null == e && (e = -1 / 0);
                    for (var n, a = 0; a < r.length; a++) {
                        var o = r[a],
                            s = u._layers[o];
                        if (!s.__builtin__ && o > e && o < t) { n = s; break }
                    }
                    n && n.renderToCanvas && (i.save(), n.renderToCanvas(i), i.restore())
                }
                if (e = e || {}, this._singleCanvas) return this._layers[0].dom;
                var r = document.createElement("canvas"),
                    n = e.pixelRatio || this.dpr;
                r.width = this.getWidth() * n, r.height = this.getHeight() * n;
                var i = r.getContext("2d");
                i.dpr = n, i.clearRect(0, 0, r.width, r.height), e.backgroundColor && (i.fillStyle = e.backgroundColor, i.fillRect(0, 0, r.width, r.height));
                for (var a, o = this.storage.getDisplayList(!0), s = {}, u = this, h = { ctx: i }, l = 0; l < o.length; l++) {
                    var c = o[l];
                    c.zlevel !== a && (t(a, c.zlevel), a = c.zlevel), this._doPaintEl(c, h, !0, s)
                }
                return t(a, 1 / 0), r
            }, e
        }, s.registerPostUpdate(function(e, t) {
            var r = t.getZr();
            (r.__egl = r.__egl || new a(r)).update(e, t)
        }), s.registerPreprocessor(r(168)), s.graphicGL = r(2), e.exports = a
    }, function(e, t, r) { r(98), r(96), r(94), r(95), r(97), r(85), r(88), r(91), r(89), r(93), r(90), r(92), r(87), r(86) }, function(e, t, r) {
        var n = r(0),
            i = r(33),
            a = r(24),
            o = r(29),
            s = n.extendSeriesModel({
                type: "series.bar3D",
                dependencies: ["globe"],
                visualColorAccessPath: "itemStyle.color",
                getInitialData: function(e, t) {
                    var r = n.getCoordinateSystemDimensions(this.get("coordinateSystem")) || ["x", "y", "z"],
                        i = n.helper.completeDimensions(r, e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") });
                    i.forEach(function(e) { e.coordDim === r[2] && (e.stackable = !0) });
                    var a = new n.List(i, this);
                    return a.initData(e.data), a
                },
                getFormattedLabel: function(e, t, r, n) { var i = a.getFormattedLabel(this, e, t, r, n); return null == i && (i = this.getData().get("z", e)), i },
                formatTooltip: function(e) { return o(this, e) },
                defaultOption: { coordinateSystem: "cartesian3D", globeIndex: 0, grid3DIndex: 0, zlevel: -10, bevelSize: 0, bevelSmoothness: 2, onGridPlane: "xy", shading: "color", minHeight: 0, itemStyle: { opacity: 1 }, label: { show: !1, distance: 2, textStyle: { fontSize: 14, color: "#000", backgroundColor: "rgba(255,255,255,0.7)", padding: 3, borderRadius: 3 } }, emphasis: { label: { show: !0 } }, animationDurationUpdate: 500 }
            });
        n.util.merge(s.prototype, i), e.exports = s
    }, function(e, t, r) {
        function n(e) { var t = a.createShader("ecgl." + e); return t.define("both", "VERTEX_COLOR"), t }
        var i = r(0),
            a = r(2),
            o = r(4),
            s = r(24),
            u = r(173),
            h = r(49),
            l = r(1).vec3;
        e.exports = i.extendChartView({
            type: "bar3D",
            __ecgl__: !0,
            init: function(e, t) {
                this.groupGL = new a.Node;
                var r = {};
                a.COMMON_SHADERS.forEach(function(e) { r[e] = new a.Material({ shader: n(e) }) }), this._materials = r, this._api = t, this._labelsBuilder = new h(256, 256, t);
                var i = this;
                this._labelsBuilder.getLabelPosition = function(e, t, r) {
                    if (i._data) {
                        var n = i._data.getItemLayout(e),
                            a = n[0],
                            o = n[1],
                            s = n[2][1];
                        return l.scaleAndAdd([], a, o, r + s)
                    }
                    return [0, 0]
                }, this._labelsBuilder.getMesh().renderOrder = 100
            },
            render: function(e, t, r) {
                var n = this._prevBarMesh;
                this._prevBarMesh = this._barMesh, this._barMesh = n, this._barMesh || (this._barMesh = new a.Mesh({ geometry: new u, shadowDepthMaterial: new a.Material({ shader: new a.Shader({ vertex: a.Shader.source("ecgl.sm.depth.vertex"), fragment: a.Shader.source("ecgl.sm.depth.fragment") }) }), culling: "cartesian3D" === e.coordinateSystem.type, renderOrder: 10, renderNormal: !0 })), this.groupGL.remove(this._prevBarMesh), this.groupGL.add(this._barMesh), this.groupGL.add(this._labelsBuilder.getMesh());
                var i = e.coordinateSystem;
                if (this._doRender(e, r), i && i.viewGL) {
                    i.viewGL.add(this.groupGL);
                    var o = i.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._barMesh.material.shader[o]("fragment", "SRGB_DECODE")
                }
                this._data = e.getData(), this._labelsBuilder.updateData(this._data), this._labelsBuilder.updateLabels(), this._updateAnimation(e)
            },
            _updateAnimation: function(e) {
                a.updateVertexAnimation([
                    ["prevPosition", "position"],
                    ["prevNormal", "normal"]
                ], this._prevBarMesh, this._barMesh, e)
            },
            _doRender: function(e, t) {
                var r = e.getData(),
                    n = e.get("shading"),
                    i = "color" !== n,
                    o = this,
                    s = this._barMesh;
                this._materials[n] ? s.material = this._materials[n] : s.material = this._materials.lambert, a.setMaterialFromModel(n, s.material, e, t), s.geometry.enableNormal = i, s.geometry.resetOffset();
                var u = e.get("bevelSize"),
                    h = e.get("bevelSmoothness");
                s.geometry.bevelSegments = h, s.geometry.bevelSize = u;
                var l = [],
                    c = new Float32Array(4 * r.count()),
                    d = 0,
                    f = 0,
                    p = !1;
                r.each(function(e) {
                    if (r.hasValue(e)) {
                        var t = r.getItemVisual(e, "color"),
                            n = r.getItemVisual(e, "opacity");
                        null == n && (n = 1), a.parseColor(t, l), l[3] *= n, c[d++] = l[0], c[d++] = l[1], c[d++] = l[2], c[d++] = l[3], l[3] > 0 && (f++, l[3] < .99 && (p = !0))
                    }
                }), s.geometry.setBarCount(f);
                var _ = r.getLayout("orient"),
                    m = this._barIndexOfData = new Int32Array(r.count()),
                    f = 0;
                r.each(function(e) {
                    if (!r.hasValue(e)) return void(m[e] = -1);
                    var t = r.getItemLayout(e),
                        n = t[0],
                        i = t[1],
                        a = t[2],
                        s = 4 * e;
                    l[0] = c[s++], l[1] = c[s++], l[2] = c[s++], l[3] = c[s++], l[3] > 0 && (o._barMesh.geometry.addBar(n, i, _, a, l, e), m[e] = f++)
                }), s.geometry.dirty(), s.geometry.updateBoundingBox();
                var g = s.material;
                g.transparent = p, g.depthMask = !p, s.geometry.sortTriangles = p, this._initHandler(e, t)
            },
            _initHandler: function(e, t) {
                var r = e.getData(),
                    n = this._barMesh,
                    i = "cartesian3D" === e.coordinateSystem.type;
                n.seriesIndex = e.seriesIndex;
                var a = -1;
                n.off("mousemove"), n.off("mouseout"), n.on("mousemove", function(e) {
                    var o = n.geometry.getDataIndexOfVertex(e.triangle[0]);
                    o !== a && (this._downplay(a), this._highlight(o), this._labelsBuilder.updateLabels([o]), i && t.dispatchAction({ type: "grid3DShowAxisPointer", value: [r.get("x", o), r.get("y", o), r.get("z", o, !0)] })), a = o, n.dataIndex = o
                }, this), n.on("mouseout", function(e) { this._downplay(a), this._labelsBuilder.updateLabels(), a = -1, n.dataIndex = -1, i && t.dispatchAction({ type: "grid3DHideAxisPointer" }) }, this)
            },
            _highlight: function(e) {
                var t = this._data;
                if (t) {
                    var r = this._barIndexOfData[e];
                    if (!(r < 0)) {
                        var n = t.getItemModel(e),
                            o = n.getModel("emphasis.itemStyle"),
                            s = o.get("color"),
                            u = o.get("opacity");
                        if (null == s) {
                            var h = t.getItemVisual(e, "color");
                            s = i.color.lift(h, -.4)
                        }
                        null == u && (u = t.getItemVisual(e, "opacity"));
                        var l = a.parseColor(s);
                        l[3] *= u, this._barMesh.geometry.setColor(r, l), this._api.getZr().refresh()
                    }
                }
            },
            _downplay: function(e) {
                var t = this._data;
                if (t) {
                    var r = this._barIndexOfData[e];
                    if (!(r < 0)) {
                        var n = t.getItemVisual(e, "color"),
                            i = t.getItemVisual(e, "opacity"),
                            o = a.parseColor(n);
                        o[3] *= i, this._barMesh.geometry.setColor(r, o), this._api.getZr().refresh()
                    }
                }
            },
            highlight: function(e, t, r, n) { this._toggleStatus("highlight", e, t, r, n) },
            downplay: function(e, t, r, n) { this._toggleStatus("downplay", e, t, r, n) },
            _toggleStatus: function(e, t, r, n, a) {
                var u = t.getData(),
                    h = o.queryDataIndex(u, a),
                    l = this;
                null != h ? i.util.each(s.normalizeToArray(h), function(t) { "highlight" === e ? this._highlight(t) : this._downplay(t) }, this) : u.each(function(t) { "highlight" === e ? l._highlight(t) : l._downplay(t) })
            },
            remove: function() { this.groupGL.removeAll() },
            dispose: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        function n(e, t) {
            var r = e.getData(),
                n = e.get("minHeight") || 0,
                i = e.get("barSize"),
                a = ["lng", "lat", "alt"].map(function(t) { return e.coordDimToDataDim(t)[0] });
            if (null == i) {
                var h = t.radius * Math.PI,
                    c = l(r, a[0], a[1]);
                i = [h / Math.sqrt(r.count() / c), h / Math.sqrt(r.count() / c)]
            } else o.util.isArray(i) || (i = [i, i]);
            r.each(a, function(e, o, s, h) {
                var l = r.get(a[2], h, !0),
                    c = r.stackedOn ? l - s : t.altitudeAxis.scale.getExtent()[0],
                    d = Math.max(t.altitudeAxis.dataToCoord(s), n),
                    f = t.dataToPoint([e, o, c]),
                    p = t.dataToPoint([e, o, l]),
                    _ = u.sub([], p, f);
                u.normalize(_, _);
                var m = [i[0], d, i[1]];
                r.setItemLayout(h, [f, _, m])
            }), r.setLayout("orient", s.UP._array)
        }

        function i(e, t) {
            var r = e.getData(),
                n = e.get("barSize"),
                i = e.get("minHeight") || 0,
                a = ["lng", "lat", "alt"].map(function(t) { return e.coordDimToDataDim(t)[0] });
            if (null == n) {
                var s = Math.min(t.size[0], t.size[2]),
                    u = l(r, a[0], a[1]);
                n = [s / Math.sqrt(r.count() / u), s / Math.sqrt(r.count() / u)]
            } else o.util.isArray(n) || (n = [n, n]);
            var h = [0, 1, 0];
            r.each(a, function(e, o, s, u) {
                var l = r.get(a[2], u, !0),
                    c = r.stackedOn ? l - s : t.altitudeAxis.scale.getExtent()[0],
                    d = Math.max(t.altitudeAxis.dataToCoord(s), i),
                    f = t.dataToPoint([e, o, c]),
                    p = [n[0], d, n[1]];
                r.setItemLayout(u, [f, h, p])
            }), r.setLayout("orient", [1, 0, 0])
        }

        function a(e, t) {
            var r = e.getData(),
                n = e.coordDimToDataDim("lng")[0],
                i = e.coordDimToDataDim("lat")[0],
                a = e.coordDimToDataDim("alt")[0],
                s = e.get("barSize"),
                u = e.get("minHeight") || 0;
            if (null == s) {
                var h = r.getDataExtent(n),
                    c = r.getDataExtent(i),
                    d = t.dataToPoint([h[0], c[0]]),
                    f = t.dataToPoint([h[1], c[1]]),
                    p = Math.min(Math.abs(d[0] - f[0]), Math.abs(d[1] - f[1])) || 1,
                    _ = l(r, n, i);
                s = [p / Math.sqrt(r.count() / _), p / Math.sqrt(r.count() / _)]
            } else o.util.isArray(s) || (s = [s, s]), s[0] /= t.getScale() / 16, s[1] /= t.getScale() / 16;
            var m = [0, 0, 1];
            r.each([n, i, a], function(e, n, i, o) {
                var h = r.get(a, o, !0),
                    l = r.stackedOn ? h - i : 0,
                    c = t.dataToPoint([e, n, l]),
                    d = t.dataToPoint([e, n, h]),
                    f = Math.max(d[2] - c[2], u),
                    p = [s[0], f, s[1]];
                r.setItemLayout(o, [c, m, p])
            }), r.setLayout("orient", [1, 0, 0])
        }
        var o = r(0),
            s = r(3),
            u = r(1).vec3,
            h = r(103),
            l = r(104);
        o.registerLayout(function(e, t) {
            e.eachSeriesByType("bar3D", function(e) {
                var t = e.coordinateSystem,
                    r = t && t.type;
                "globe" === r ? n(e, t) : "cartesian3D" === r ? h(e, t) : "geo3D" === r ? i(e, t) : "mapbox" === r && a(e, t)
            })
        })
    }, function(e, t, r) {
        function n(e) {
            var t = e[0],
                r = e[1];
            return !(t > 0 && r > 0 || t < 0 && r < 0)
        }

        function i(e, t) {
            var r = e.getData(),
                i = e.get("barSize");
            if (null == i) {
                var s, u, h = t.size,
                    l = t.getAxis("x"),
                    c = t.getAxis("y");
                s = "category" === l.type ? .7 * l.getBandWidth() : .6 * Math.round(h[0] / Math.sqrt(r.count())), u = "category" === c.type ? .7 * c.getBandWidth() : .6 * Math.round(h[1] / Math.sqrt(r.count())), i = [s, u]
            } else a.util.isArray(i) || (i = [i, i]);
            var d = t.getAxis("z").scale.getExtent(),
                f = n(d),
                p = ["x", "y", "z"].map(function(t) { return e.coordDimToDataDim(t)[0] });
            r.each(p, function(e, n, a, s) {
                var u = r.get(p[2], s, !0),
                    h = r.stackedOn ? u - a : f ? 0 : d[0],
                    l = t.dataToPoint([e, n, h]),
                    c = t.dataToPoint([e, n, u]),
                    _ = o.dist(l, c),
                    m = [0, c[1] < l[1] ? -1 : 1, 0];
                0 === Math.abs(_) && (_ = .1);
                var g = [i[0], _, i[1]];
                r.setItemLayout(s, [l, m, g])
            }), r.setLayout("orient", [1, 0, 0])
        }
        var a = r(0),
            o = r(1).vec3;
        e.exports = i
    }, function(e, t) {
        e.exports = function(e, t, r) {
            for (var n = e.getDataExtent(t), i = e.getDataExtent(r), a = n[1] - n[0] || n[0], o = i[1] - i[0] || i[0], s = new Uint8Array(2500), u = 0; u < e.count(); u++) {
                var h = e.get(t, u),
                    l = e.get(r, u),
                    c = Math.floor((h - n[0]) / a * 49),
                    d = Math.floor((l - i[0]) / o * 49),
                    f = 50 * d + c;
                s[f] = s[f] || 1
            }
            for (var p = 0, u = 0; u < s.length; u++) s[u] && p++;
            return p / s.length
        }
    }, function(e, t, r) {
        var n = r(2),
            i = r(177),
            a = r(0),
            o = r(1),
            s = o.vec4;
        n.Shader.import(r(106));
        var u = n.Mesh.extend(function() {
            var e = new n.Geometry({ dynamic: !0, attributes: { color: new n.Geometry.Attribute("color", "float", 4, "COLOR"), position: new n.Geometry.Attribute("position", "float", 3, "POSITION"), size: new n.Geometry.Attribute("size", "float", 1), prevPosition: new n.Geometry.Attribute("prevPosition", "float", 3), prevSize: new n.Geometry.Attribute("prevSize", "float", 1) } });
            a.util.extend(e, i);
            var t = new n.Material({ shader: n.createShader("ecgl.sdfSprite"), transparent: !0, depthMask: !1 });
            t.shader.enableTexture("sprite"), t.shader.define("both", "VERTEX_COLOR");
            var r = new n.Texture2D({ image: document.createElement("canvas"), flipY: !1 });
            return t.set("sprite", r), e.pick = this._pick.bind(this), { geometry: e, material: t, mode: n.Mesh.POINTS, sizeScale: 1 }
        }, {
            _pick: function(e, t, r, i, a, o) {
                var s = this._positionNDC;
                if (s)
                    for (var u = r.viewport, h = 2 / u.width, l = 2 / u.height, c = this.geometry.vertexCount - 1; c >= 0; c--) {
                        var d;
                        d = this.geometry.indices ? this.geometry.indices[c] : c;
                        var f = s[2 * d],
                            p = s[2 * d + 1],
                            _ = this.geometry.attributes.size.get(d) / this.sizeScale,
                            m = _ / 2;
                        if (e > f - m * h && e < f + m * h && t > p - m * l && t < p + m * l) {
                            var g = new n.Vector3,
                                v = new n.Vector3;
                            this.geometry.attributes.position.get(d, g._array), n.Vector3.transformMat4(v, g, this.worldTransform), o.push({ vertexIndex: d, point: g, pointWorld: v, target: this, distance: v.distance(i.getWorldPosition()) })
                        }
                    }
            },
            updateNDCPosition: function(e, t, r) {
                var n = this._positionNDC,
                    i = this.geometry;
                n && n.length / 2 === i.vertexCount || (n = this._positionNDC = new Float32Array(2 * i.vertexCount));
                for (var a = s.create(), o = 0; o < i.vertexCount; o++) i.attributes.position.get(o, a), a[3] = 1, s.transformMat4(a, a, e._array), s.scale(a, a, 1 / a[3]), n[2 * o] = a[0], n[2 * o + 1] = a[1]
            }
        });
        e.exports = u
    }, function(e, t) { e.exports = "@export ecgl.sdfSprite.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform float elapsedTime : 0;\n\nattribute vec3 position : POSITION;\nattribute float size;\n\n#ifdef VERTEX_COLOR\nattribute vec4 a_FillColor: COLOR;\nvarying vec4 v_Color;\n#endif\n\n#ifdef VERTEX_ANIMATION\nattribute vec3 prevPosition;\nattribute float prevSize;\nuniform float percent : 1.0;\n#endif\n\n\n#ifdef POSITIONTEXTURE_ENABLED\nuniform sampler2D positionTexture;\n#endif\n\nvarying float v_Size;\n\nvoid main()\n{\n\n#ifdef POSITIONTEXTURE_ENABLED\n gl_Position = worldViewProjection * vec4(texture2D(positionTexture, position.xy).xy, -10.0, 1.0);\n#else\n\n #ifdef VERTEX_ANIMATION\n vec3 pos = mix(prevPosition, position, percent);\n #else\n vec3 pos = position;\n #endif\n gl_Position = worldViewProjection * vec4(pos, 1.0);\n#endif\n\n#ifdef VERTEX_ANIMATION\n v_Size = mix(prevSize, size, percent);\n#else\n v_Size = size;\n#endif\n\n#ifdef VERTEX_COLOR\n v_Color = a_FillColor;\n #endif\n\n gl_PointSize = v_Size;\n}\n\n@end\n\n@export ecgl.sdfSprite.fragment\n\nuniform vec4 color: [1, 1, 1, 1];\nuniform vec4 strokeColor: [1, 1, 1, 1];\nuniform float smoothing: 0.07;\n\nuniform float lineWidth: 0.0;\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\n\nvarying float v_Size;\n\nuniform sampler2D sprite;\n\n@import qtek.util.srgb\n\nvoid main()\n{\n gl_FragColor = color;\n\n vec4 _strokeColor = strokeColor;\n\n#ifdef VERTEX_COLOR\n gl_FragColor *= v_Color;\n #endif\n\n#ifdef SPRITE_ENABLED\n float d = texture2D(sprite, gl_PointCoord).r;\n gl_FragColor.a *= smoothstep(0.5 - smoothing, 0.5 + smoothing, d);\n\n if (lineWidth > 0.0) {\n float sLineWidth = lineWidth / 2.0;\n\n float outlineMaxValue0 = 0.5 + sLineWidth;\n float outlineMaxValue1 = 0.5 + sLineWidth + smoothing;\n float outlineMinValue0 = 0.5 - sLineWidth - smoothing;\n float outlineMinValue1 = 0.5 - sLineWidth;\n\n if (d <= outlineMaxValue1 && d >= outlineMinValue0) {\n float a = _strokeColor.a;\n if (d <= outlineMinValue1) {\n a = a * smoothstep(outlineMinValue0, outlineMinValue1, d);\n }\n else {\n a = a * smoothstep(outlineMaxValue1, outlineMaxValue0, d);\n }\n gl_FragColor.rgb = mix(gl_FragColor.rgb * gl_FragColor.a, _strokeColor.rgb, a);\n gl_FragColor.a = gl_FragColor.a * (1.0 - a) + a;\n }\n }\n#endif\n\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(gl_FragColor);\n#endif\n}\n@end" }, function(e, t, r) {
        var n = r(0);
        n.extendSeriesModel({
            type: "series.flowGL",
            dependencies: ["geo", "grid", "bmap"],
            visualColorAccessPath: "itemStyle.color",
            getInitialData: function(e, t) {
                var r = n.getCoordinateSystemDimensions(this.get("coordinateSystem")) || ["x", "y"];
                r.push("vx", "vy");
                var i = n.helper.completeDimensions(r, e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") }),
                    a = new n.List(i, this);
                return a.initData(e.data), a
            },
            defaultOption: { coordinateSystem: "cartesian2d", zlevel: 10, particleDensity: 128, particleSize: 1, particleSpeed: 1, particleTrail: 2, colorTexture: null, gridWidth: "auto", gridHeight: "auto", itemStyle: { color: "#fff", opacity: .8 } }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(4),
            o = r(21),
            s = r(109);
        n.extendChartView({
            type: "flowGL",
            __ecgl__: !0,
            init: function(e, t) {
                this.viewGL = new o("orthographic"), this.groupGL = new i.Node, this.viewGL.add(this.groupGL), this._particleSurface = new s;
                var r = new i.Mesh({ geometry: new i.PlaneGeometry, material: new i.Material({ shader: new i.Shader({ vertex: i.Shader.source("ecgl.color.vertex"), fragment: i.Shader.source("ecgl.color.fragment") }), transparent: !0 }) });
                r.material.shader.enableTexture("diffuseMap"), this.groupGL.add(r), this._planeMesh = r
            },
            render: function(e, t, r) {
                var n = this._particleSurface;
                this._updateData(e, r), this._updateCamera(r.getWidth(), r.getHeight(), r.getDevicePixelRatio());
                var o = a.firstNotNull(e.get("particleDensity"), 128);
                n.setParticleDensity(o, o);
                var s = this._planeMesh,
                    u = +new Date,
                    h = this;
                s.__percent = 0, s.stopAnimation(), s.animate("", { loop: !0 }).when(1e5, { __percent: 1 }).during(function() {
                    var e = +new Date,
                        t = e - u;
                    u = e, h._renderer && (n.update(h._renderer, t / 1e3), s.material.set("diffuseMap", n.getSurfaceTexture()))
                }).start();
                var l = e.getModel("itemStyle"),
                    c = i.parseColor(l.get("color"));
                c[3] *= a.firstNotNull(l.get("opacity"), 1), s.material.set("color", c), this._particleSurface.setColorTextureImage(e.get("colorTexture"), r), this._particleSurface.particleSize = e.get("particleSize"), this._particleSurface.particleSpeedScaling = e.get("particleSpeed"), this._particleSurface.motionBlurFactor = 1 - Math.pow(.1, e.get("particleTrail"))
            },
            updateLayout: function(e, t, r) { this._updateData(e, r) },
            afterRender: function(e, t, r, n) {
                var i = n.renderer;
                this._renderer = i
            },
            _updateData: function(e, t) {
                var r = e.coordinateSystem,
                    n = r.dimensions.map(function(t) { return e.coordDimToDataDim(t)[0] }),
                    i = e.getData(),
                    a = i.getDataExtent(n[0]),
                    o = i.getDataExtent(n[1]),
                    s = e.get("gridWidth"),
                    u = e.get("gridHeight");
                if (null == s || "auto" === s) {
                    var h = (a[1] - a[0]) / (o[1] - o[0]);
                    s = Math.round(Math.sqrt(h * i.count()))
                }
                null != u && "auto" !== u || (u = Math.ceil(i.count() / s));
                var l = this._particleSurface.vectorFieldTexture,
                    c = l.pixels;
                if (c && c.length === u * s * 4)
                    for (var d = 0; d < c.length; d++) c[d] = 0;
                else c = l.pixels = new Float32Array(s * u * 4);
                var f = 0,
                    p = 1 / 0,
                    _ = new Float32Array(2 * i.count()),
                    m = 0,
                    g = [
                        [1 / 0, 1 / 0],
                        [-1 / 0, -1 / 0]
                    ];
                i.each([n[0], n[1], "vx", "vy"], function(e, t, n, i) {
                    var a = r.dataToPoint([e, t]);
                    _[m++] = a[0], _[m++] = a[1], g[0][0] = Math.min(a[0], g[0][0]), g[0][1] = Math.min(a[1], g[0][1]), g[1][0] = Math.max(a[0], g[1][0]), g[1][1] = Math.max(a[1], g[1][1]);
                    var o = Math.sqrt(n * n + i * i);
                    f = Math.max(f, o), p = Math.min(p, o)
                }), i.each(["vx", "vy"], function(e, t, r) {
                    var n = Math.round((_[2 * r] - g[0][0]) / (g[1][0] - g[0][0]) * (s - 1)),
                        i = u - 1 - Math.round((_[2 * r + 1] - g[0][1]) / (g[1][1] - g[0][1]) * (u - 1)),
                        a = 4 * (i * s + n);
                    c[a] = e / f * .5 + .5, c[a + 1] = t / f * .5 + .5, c[a + 3] = 1
                }), l.width = s, l.height = u, "bmap" === e.get("coordinateSystem") && this._fillEmptyPixels(l), l.dirty(), this._updatePlanePosition(g[0], g[1], t), this._updateGradientTexture(i.getVisual("visualMeta"), [p, f])
            },
            _fillEmptyPixels: function(e) {
                function t(e, t, r) { e = Math.max(Math.min(e, i - 1), 0), t = Math.max(Math.min(t, a - 1), 0); var o = 4 * (t * (i - 1) + e); return 0 !== n[o + 3] && (r[0] = n[o], r[1] = n[o + 1], !0) }

                function r(e, t, r) { r[0] = e[0] + t[0], r[1] = e[1] + t[1] }
                for (var n = e.pixels, i = e.width, a = e.height, o = [], s = [], u = [], h = [], l = [], c = 0, d = 0; d < a; d++)
                    for (var f = 0; f < i; f++) {
                        var p = 4 * (d * (i - 1) + f);
                        0 === n[p + 3] && (c = o[0] = o[1] = 0, t(f - 1, d, s) && (c++, r(s, o, o)), t(f + 1, d, u) && (c++, r(u, o, o)), t(f, d - 1, h) && (c++, r(h, o, o)), t(f, d + 1, l) && (c++, r(l, o, o)), o[0] /= c, o[1] /= c, n[p] = o[0], n[p + 1] = o[1]), n[p + 3] = 1
                    }
            },
            _updateGradientTexture: function(e, t) {
                if (!e || !e.length) return void this._particleSurface.setGradientTexture(null);
                this._gradientTexture = this._gradientTexture || new i.Texture2D({ image: document.createElement("canvas") });
                var r = this._gradientTexture,
                    n = r.image;
                n.width = 200, n.height = 1;
                var a = n.getContext("2d"),
                    o = a.createLinearGradient(0, .5, n.width, .5);
                e[0].stops.forEach(function(e) {
                    var r;
                    t[1] === t[0] ? r = 0 : (r = e.value / t[1], r = Math.min(Math.max(r, 0), 1)), o.addColorStop(r, e.color)
                }), a.fillStyle = o, a.fillRect(0, 0, n.width, n.height), r.dirty(), this._particleSurface.setGradientTexture(this._gradientTexture)
            },
            _updatePlanePosition: function(e, t, r) {
                var n = this._limitInViewport(e, t, r);
                e = n.leftTop, t = n.rightBottom, this._particleSurface.setRegion(n.region), this._planeMesh.position.set((e[0] + t[0]) / 2, r.getHeight() - (e[1] + t[1]) / 2, 0);
                var i = t[0] - e[0],
                    a = t[1] - e[1];
                this._planeMesh.scale.set(i / 2, a / 2, 1), this._particleSurface.resize(Math.min(i, 2048), Math.min(a, 2048)), this._renderer && this._particleSurface.clearFrame(this._renderer)
            },
            _limitInViewport: function(e, t, r) {
                var n = [Math.max(e[0], 0), Math.max(e[1], 0)],
                    i = [Math.min(t[0], r.getWidth()), Math.min(t[1], r.getHeight())],
                    a = t[0] - e[0],
                    o = t[1] - e[1],
                    s = i[0] - n[0],
                    u = i[1] - n[1];
                return { leftTop: n, rightBottom: i, region: [(n[0] - e[0]) / a, 1 - u / o - (n[1] - e[1]) / o, s / a, u / o] }
            },
            _updateCamera: function(e, t, r) {
                this.viewGL.setViewport(0, 0, e, t, r);
                var n = this.viewGL.camera;
                n.left = n.bottom = 0, n.top = t, n.right = e, n.near = 0, n.far = 100, n.position.z = 10
            },
            remove: function() { this._planeMesh.stopAnimation(), this.groupGL.removeAll() },
            dispose: function() { this._renderer && this._particleSurface.dispose(this._renderer), this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        var n = r(12),
            i = r(13),
            a = r(25),
            o = r(16),
            s = r(7),
            u = r(5),
            h = r(6),
            l = r(36),
            c = r(26),
            d = r(46),
            f = r(10);
        s.import(r(110));
        var p = function() { this.motionBlurFactor = .99, this.vectorFieldTexture = new u({ type: h.FLOAT, flipY: !1 }), this.particleLife = [5, 20], this.particleSize = 1, this.particleColor = [1, 1, 1, 1], this.particleSpeedScaling = 1, this._thisFrameTexture = null, this._particlePass = null, this._spawnTexture = null, this._particleTexture0 = null, this._particleTexture1 = null, this._particleMesh = null, this._surfaceFrameBuffer = null, this._elapsedTime = 0, this._scene = null, this._camera = null, this._lastFrameTexture = null, this.init() };
        p.prototype = {
            constructor: p,
            init: function() {
                var e = { type: h.FLOAT, minFilter: h.NEAREST, magFilter: h.NEAREST, wrapS: h.REPEAT, wrapT: h.REPEAT, useMipmap: !1 };
                this._spawnTexture = new u(e), this._particleTexture0 = new u(e), this._particleTexture1 = new u(e), this._frameBuffer = new f({ depthBuffer: !1 }), this._particlePass = new n({ fragment: s.source("ecgl.vfParticle.particle.fragment") }), this._particlePass.setUniform("velocityTexture", this.vectorFieldTexture), this._particlePass.setUniform("spawnTexture", this._spawnTexture);
                var t = new a({ renderOrder: 10, material: new o({ shader: new s({ vertex: s.source("ecgl.vfParticle.renderPoints.vertex"), fragment: s.source("ecgl.vfParticle.renderPoints.fragment") }) }), mode: a.POINTS, geometry: new i({ mainAttribute: "texcoord0" }) }),
                    r = new a({ material: new o({ shader: new s({ vertex: s.source("ecgl.color.vertex"), fragment: s.source("ecgl.color.fragment") }) }), geometry: new d });
                r.material.shader.enableTexture("diffuseMap"), this._particleMesh = t, this._lastFrameFullQuadMesh = r, this._scene = new c, this._scene.add(this._particleMesh), this._scene.add(r), this._camera = new l, this._thisFrameTexture || (this._thisFrameTexture = new u({ width: 1024, height: 1024 }));
                var p = this._thisFrameTexture.width,
                    _ = this._thisFrameTexture.height;
                this._lastFrameTexture = new u({ width: p, height: _ })
            },
            setParticleDensity: function(e, t) {
                var r = this._particleMesh.geometry,
                    n = e * t,
                    i = r.attributes;
                i.texcoord0.init(n);
                for (var a = new Float32Array(4 * n), o = 0, s = this.particleLife, u = 0; u < e; u++)
                    for (var h = 0; h < t; h++, o++) {
                        i.texcoord0.value[2 * o] = u / e, i.texcoord0.value[2 * o + 1] = h / t, a[4 * o] = Math.random(), a[4 * o + 1] = Math.random(), a[4 * o + 2] = Math.random();
                        var l = (s[1] - s[0]) * Math.random() + s[0];
                        a[4 * o + 3] = l
                    }
                this._spawnTexture.width = e, this._spawnTexture.height = t, this._spawnTexture.pixels = a, this._particleTexture0.width = this._particleTexture1.width = e, this._particleTexture0.height = this._particleTexture1.height = t
            },
            update: function(e, t) {
                var r = this._particleMesh,
                    n = this._frameBuffer,
                    i = this._particlePass;
                r.material.set("size", this.particleSize * e.getDevicePixelRatio()), r.material.set("color", this.particleColor), i.setUniform("speedScaling", this.particleSpeedScaling), n.attach(this._particleTexture1), i.setUniform("particleTexture", this._particleTexture0), i.setUniform("deltaTime", t), i.setUniform("elapsedTime", this._elapsedTime), i.render(e, n), r.material.set("particleTexture", this._particleTexture1), n.attach(this._thisFrameTexture), n.bind(e), e.gl.clear(e.gl.DEPTH_BUFFER_BIT | e.gl.COLOR_BUFFER_BIT), this._lastFrameFullQuadMesh.material.set("diffuseMap", this._lastFrameTexture), this._lastFrameFullQuadMesh.material.set("color", [1, 1, 1, this.motionBlurFactor]), e.render(this._scene, this._camera), n.unbind(e), this._swapTexture(), this._elapsedTime += t
            },
            getSurfaceTexture: function() { return this._lastFrameTexture },
            setRegion: function(e) { this._particlePass.setUniform("region", e) },
            resize: function(e, t) { this._lastFrameTexture.width = e, this._lastFrameTexture.height = t, this._thisFrameTexture.width = e, this._thisFrameTexture.height = t },
            setGradientTexture: function(e) {
                var t = this._particleMesh.material;
                t.shader[e ? "enableTexture" : "disableTexture"]("gradientTexture"), t.setUniform("gradientTexture", e)
            },
            setColorTextureImage: function(e, t) { this._particleMesh.material.setTextureImage("colorTexture", e, t, { flipY: !0 }) },
            clearFrame: function(e) {
                var t = this._frameBuffer;
                t.attach(this._lastFrameTexture), t.bind(e), e.gl.clear(e.gl.DEPTH_BUFFER_BIT | e.gl.COLOR_BUFFER_BIT), t.unbind(e)
            },
            _swapTexture: function() {
                var e = this._particleTexture0;
                this._particleTexture0 = this._particleTexture1, this._particleTexture1 = e;
                var e = this._thisFrameTexture;
                this._thisFrameTexture = this._lastFrameTexture, this._lastFrameTexture = e
            },
            dispose: function(e) { e.disposeFrameBuffer(this._frameBuffer), e.disposeTexture(this.vectorFieldTexture), e.disposeTexture(this._spawnTexture), e.disposeTexture(this._particleTexture0), e.disposeTexture(this._particleTexture1), e.disposeTexture(this._thisFrameTexture), e.disposeTexture(this._lastFrameTexture), e.disposeScene(this._scene) }
        }, e.exports = p
    }, function(e, t) { e.exports = "@export ecgl.vfParticle.particle.fragment\n\nuniform sampler2D particleTexture;\nuniform sampler2D spawnTexture;\nuniform sampler2D velocityTexture;\n\nuniform float deltaTime;\nuniform float elapsedTime;\n\nuniform float speedScaling : 1.0;\n\nuniform vec4 region : [0, 0, 1, 1];\n\nvarying vec2 v_Texcoord;\n\nvoid main()\n{\n vec4 p = texture2D(particleTexture, v_Texcoord);\n if (p.w > 0.0) {\n vec4 vTex = texture2D(velocityTexture, p.xy * region.zw + region.xy);\n vec2 v = vTex.xy;\n v = (v - 0.5) * 2.0;\n p.z = length(v);\n p.xy += v * deltaTime / 10.0 * speedScaling;\n p.xy = fract(p.xy);\n p.w -= deltaTime;\n }\n else {\n p = texture2D(spawnTexture, fract(v_Texcoord + elapsedTime / 10.0));\n p.z = 0.0;\n }\n gl_FragColor = p;\n}\n@end\n\n@export ecgl.vfParticle.renderPoints.vertex\n\n#define PI 3.1415926\n\nattribute vec2 texcoord : TEXCOORD_0;\n\nuniform sampler2D particleTexture;\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nuniform float size : 1.0;\n\nvarying float v_Mag;\nvarying vec2 v_Uv;\n\nvoid main()\n{\n vec4 p = texture2D(particleTexture, texcoord);\n\n if (p.w > 0.0 && p.z > 1e-5) {\n gl_Position = worldViewProjection * vec4(p.xy * 2.0 - 1.0, 0.0, 1.0);\n }\n else {\n gl_Position = vec4(100000.0, 100000.0, 100000.0, 1.0);\n }\n\n v_Mag = p.z;\n v_Uv = p.xy;\n\n gl_PointSize = size;\n}\n\n@end\n\n@export ecgl.vfParticle.renderPoints.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\nuniform sampler2D gradientTexture;\nuniform sampler2D colorTexture;\n\nvarying float v_Mag;\nvarying vec2 v_Uv;\n\nvoid main()\n{\n gl_FragColor = color;\n#ifdef GRADIENTTEXTURE_ENABLED\n gl_FragColor *= texture2D(gradientTexture, vec2(v_Mag, 0.5));\n#endif\n#ifdef COLORTEXTURE_ENABLED\n gl_FragColor *= texture2D(colorTexture, v_Uv);\n#endif\n}\n\n@end\n" }, function(e, t, r) {
        var n = r(5),
            i = r(6),
            a = r(117),
            o = a.toString();
        o = o.slice(o.indexOf("{") + 1, o.lastIndexOf("}"));
        var s = { barnesHutOptimize: !0, barnesHutTheta: 1.5, repulsionByDegree: !0, linLogMode: !1, strongGravityMode: !1, gravity: 1, scaling: 1, edgeWeightInfluence: 1, jitterTolerence: .1, preventOverlap: !1, dissuadeHubs: !1, gravityCenter: null },
            u = function(e) {
                for (var t in s) this[t] = s[t];
                if (e)
                    for (var t in e) this[t] = e[t];
                this._nodes = [], this._edges = [], this._disposed = !1, this._positionTex = new n({ type: i.FLOAT, flipY: !1, minFilter: i.NEAREST, magFilter: i.NEAREST })
            };
        u.prototype.initData = function(e, t) {
            var r = new Blob([o]),
                n = window.URL.createObjectURL(r);
            this._worker = new Worker(n), this._worker.onmessage = this._$onupdate.bind(this), this._nodes = e, this._edges = t, this._frame = 0;
            for (var i = e.length, a = t.length, s = new Float32Array(2 * i), u = new Float32Array(i), h = new Float32Array(i), l = new Float32Array(2 * a), c = new Float32Array(a), d = 0; d < e.length; d++) {
                var f = e[d];
                s[2 * d] = f.x, s[2 * d + 1] = f.y, u[d] = null == f.mass ? 1 : f.mass, h[d] = null == f.size ? 1 : f.size
            }
            for (var d = 0; d < t.length; d++) {
                var p = t[d],
                    _ = p.node1,
                    m = p.node2;
                l[2 * d] = _, l[2 * d + 1] = m, c[d] = null == p.weight ? 1 : p.weight
            }
            var g = Math.ceil(Math.sqrt(e.length)),
                v = g,
                y = new Float32Array(g * v * 4),
                x = this._positionTex;
            x.width = g, x.height = v, x.pixels = y, this._worker.postMessage({ cmd: "init", nodesPosition: s, nodesMass: u, nodesSize: h, edges: l, edgesWeight: c }), this._globalSpeed = 1 / 0
        }, u.prototype.updateOption = function(e) {
            var t = {};
            for (var r in s) t[r] = s[r];
            var n = this._nodes,
                i = this._edges,
                a = n.length;
            if (t.jitterTolerence = a > 5e4 ? 10 : a > 5e3 ? 1 : .1, t.scaling = a > 100 ? 2 : 10, t.barnesHutOptimize = a > 1e3, e)
                for (var r in s) null != e[r] && (t[r] = e[r]);
            if (!t.gravityCenter) {
                for (var o = [1 / 0, 1 / 0], u = [-1 / 0, -1 / 0], h = 0; h < n.length; h++) o[0] = Math.min(n[h].x, o[0]), o[1] = Math.min(n[h].y, o[1]), u[0] = Math.max(n[h].x, u[0]), u[1] = Math.max(n[h].y, u[1]);
                t.gravityCenter = [.5 * (o[0] + u[0]), .5 * (o[1] + u[1])]
            }
            for (var h = 0; h < i.length; h++) {
                var l = i[h].node1,
                    c = i[h].node2;
                n[l].degree = (n[l].degree || 0) + 1, n[c].degree = (n[c].degree || 0) + 1
            }
            this._worker && this._worker.postMessage({ cmd: "updateConfig", config: t })
        }, u.prototype.update = function(e, t, r) { null == t && (t = 1), t = Math.max(t, 1), this._frame += t, this._onupdate = r, this._worker && this._worker.postMessage({ cmd: "update", steps: Math.round(t) }) }, u.prototype._$onupdate = function(e) {
            if (!this._disposed) {
                var t = new Float32Array(e.data.buffer);
                this._globalSpeed = e.data.globalSpeed, this._positionArr = t, this._updateTexture(t), this._onupdate && this._onupdate()
            }
        }, u.prototype.getNodePositionTexture = function() { return this._positionTex }, u.prototype.getNodeUV = function(e, t) {
            t = t || [];
            var r = this._positionTex.width,
                n = this._positionTex.height;
            return t[0] = e % r / (r - 1), t[1] = Math.floor(e / r) / (n - 1), t
        }, u.prototype.getNodes = function() { return this._nodes }, u.prototype.getEdges = function() { return this._edges }, u.prototype.isFinished = function(e) { return this._globalSpeed < e && this._frame > 10 }, u.prototype.getNodePosition = function(e, t) {
            if (t || (t = new Float32Array(2 * this._nodes.length)), this._positionArr)
                for (var r = 0; r < this._positionArr.length; r++) t[r] = this._positionArr[r];
            return t
        }, u.prototype._updateTexture = function(e) {
            for (var t = this._positionTex.pixels, r = 0, n = 0; n < e.length;) t[r++] = e[n++], t[r++] = e[n++], t[r++] = 1, t[r++] = 1;
            this._positionTex.dirty()
        }, u.prototype.dispose = function(e) { this._disposed = !0, this._worker = null }, e.exports = u
    }, function(e, t, r) {
        function n(e) {
            var t = { type: a.Texture.FLOAT, minFilter: a.Texture.NEAREST, magFilter: a.Texture.NEAREST };
            this._positionSourceTex = new a.Texture2D(t), this._positionSourceTex.flipY = !1, this._positionTex = new a.Texture2D(t), this._positionPrevTex = new a.Texture2D(t), this._forceTex = new a.Texture2D(t), this._forcePrevTex = new a.Texture2D(t), this._weightedSumTex = new a.Texture2D(t), this._weightedSumTex.width = this._weightedSumTex.height = 1, this._globalSpeedTex = new a.Texture2D(t), this._globalSpeedPrevTex = new a.Texture2D(t), this._globalSpeedTex.width = this._globalSpeedTex.height = 1, this._globalSpeedPrevTex.width = this._globalSpeedPrevTex.height = 1, this._nodeRepulsionPass = new o({ fragment: a.Shader.source("ecgl.forceAtlas2.updateNodeRepulsion") }), this._positionPass = new o({ fragment: a.Shader.source("ecgl.forceAtlas2.updatePosition") }), this._globalSpeedPass = new o({ fragment: a.Shader.source("ecgl.forceAtlas2.calcGlobalSpeed") }), this._copyPass = new o({ fragment: a.Shader.source("qtek.compositor.output") });
            var r = function(e) { e.blendEquation(e.FUNC_ADD), e.blendFunc(e.ONE, e.ONE) };
            this._edgeForceMesh = new a.Mesh({ geometry: new a.Geometry({ attributes: { node1: new a.Geometry.Attribute("node1", "float", 2), node2: new a.Geometry.Attribute("node2", "float", 2), weight: new a.Geometry.Attribute("weight", "float", 1) }, dynamic: !0, mainAttribute: "node1" }), material: new a.Material({ transparent: !0, shader: a.createShader("ecgl.forceAtlas2.updateEdgeAttraction"), blend: r, depthMask: !1, depthText: !1 }), mode: a.Mesh.POINTS }), this._weightedSumMesh = new a.Mesh({ geometry: new a.Geometry({ attributes: { node: new a.Geometry.Attribute("node", "float", 2) }, dynamic: !0, mainAttribute: "node" }), material: new a.Material({ transparent: !0, shader: a.createShader("ecgl.forceAtlas2.calcWeightedSum"), blend: r, depthMask: !1, depthText: !1 }), mode: a.Mesh.POINTS }), this._framebuffer = new s({ depthBuffer: !1 }), this._dummyCamera = new a.OrthographicCamera({ left: -1, right: 1, top: 1, bottom: -1, near: 0, far: 100 }), this._globalSpeed = 0
        }
        var i = r(0),
            a = r(2),
            o = r(12),
            s = r(10);
        a.Shader.import(r(116));
        var u = { repulsionByDegree: !0, linLogMode: !1, strongGravityMode: !1, gravity: 1, scaling: 1, edgeWeightInfluence: 1, jitterTolerence: .1, preventOverlap: !1, dissuadeHubs: !1, gravityCenter: null };
        n.prototype.updateOption = function(e) {
            for (var t in u) this[t] = u[t];
            var r = this._nodes.length;
            if (this.jitterTolerence = r > 5e4 ? 10 : r > 5e3 ? 1 : .1, this.scaling = r > 100 ? 2 : 10, e)
                for (var t in u) null != e[t] && (this[t] = e[t]);
            if (this.repulsionByDegree)
                for (var n = this._positionSourceTex.pixels, i = 0; i < this._nodes.length; i++) n[4 * i + 2] = (this._nodes[i].degree || 0) + 1
        }, n.prototype._updateGravityCenter = function(e) {
            var t = this._nodes,
                r = this._edges;
            if (this.gravityCenter) this._gravityCenter = this.gravityCenter;
            else {
                for (var n = [1 / 0, 1 / 0], i = [-1 / 0, -1 / 0], a = 0; a < t.length; a++) n[0] = Math.min(t[a].x, n[0]), n[1] = Math.min(t[a].y, n[1]), i[0] = Math.max(t[a].x, i[0]), i[1] = Math.max(t[a].y, i[1]);
                this._gravityCenter = [.5 * (n[0] + i[0]), .5 * (n[1] + i[1])]
            }
            for (var a = 0; a < r.length; a++) {
                var o = r[a].node1,
                    s = r[a].node2;
                t[o].degree = (t[o].degree || 0) + 1, t[s].degree = (t[s].degree || 0) + 1
            }
        }, n.prototype.initData = function(e, t) {
            this._nodes = e, this._edges = t, this._updateGravityCenter();
            var r = Math.ceil(Math.sqrt(e.length)),
                n = r,
                i = new Float32Array(r * n * 4);
            this._resize(r, n);
            for (var a = 0, o = 0; o < e.length; o++) {
                var s = e[o];
                i[a++] = s.x || 0, i[a++] = s.y || 0, i[a++] = s.mass || 1, i[a++] = s.size || 1
            }
            this._positionSourceTex.pixels = i;
            var u = this._edgeForceMesh.geometry,
                h = t.length;
            u.attributes.node1.init(2 * h), u.attributes.node2.init(2 * h), u.attributes.weight.init(2 * h);
            for (var l = [], o = 0; o < t.length; o++) {
                var c = u.attributes,
                    d = t[o].weight;
                null == d && (d = 1), c.node1.set(o, this.getNodeUV(t[o].node1, l)), c.node2.set(o, this.getNodeUV(t[o].node2, l)), c.weight.set(o, d), c.node1.set(o + h, this.getNodeUV(t[o].node2, l)), c.node2.set(o + h, this.getNodeUV(t[o].node1, l)), c.weight.set(o + h, d)
            }
            var f = this._weightedSumMesh.geometry;
            f.attributes.node.init(e.length);
            for (var o = 0; o < e.length; o++) f.attributes.node.set(o, this.getNodeUV(o, l));
            u.dirty(), f.dirty(), this._nodeRepulsionPass.material.shader.define("fragment", "NODE_COUNT", e.length), this._nodeRepulsionPass.material.setUniform("textureSize", [r, n]), this._inited = !1, this._frame = 0
        }, n.prototype.getNodes = function() { return this._nodes }, n.prototype.getEdges = function() { return this._edges }, n.prototype.step = function(e) {
            this._inited || (this._initFromSource(e), this._inited = !0), this._frame++, this._framebuffer.attach(this._forceTex), this._framebuffer.bind(e);
            var t = this._nodeRepulsionPass;
            t.setUniform("strongGravityMode", this.strongGravityMode), t.setUniform("gravity", this.gravity), t.setUniform("gravityCenter", this._gravityCenter), t.setUniform("scaling", this.scaling), t.setUniform("preventOverlap", this.preventOverlap), t.setUniform("positionTex", this._positionPrevTex), t.render(e);
            var r = this._edgeForceMesh;
            r.material.set("linLogMode", this.linLogMode), r.material.set("edgeWeightInfluence", this.edgeWeightInfluence), r.material.set("preventOverlap", this.preventOverlap), r.material.set("positionTex", this._positionPrevTex), e.gl.enable(e.gl.BLEND), e.renderQueue([r], this._dummyCamera), this._framebuffer.attach(this._weightedSumTex), e.gl.clearColor(0, 0, 0, 0), e.gl.clear(e.gl.COLOR_BUFFER_BIT), e.gl.enable(e.gl.BLEND);
            var n = this._weightedSumMesh;
            n.material.set("positionTex", this._positionPrevTex), n.material.set("forceTex", this._forceTex), n.material.set("forcePrevTex", this._forcePrevTex), e.renderQueue([n], this._dummyCamera), this._framebuffer.attach(this._globalSpeedTex);
            var i = this._globalSpeedPass;
            i.setUniform("globalSpeedPrevTex", this._globalSpeedPrevTex), i.setUniform("weightedSumTex", this._weightedSumTex), i.setUniform("jitterTolerence", this.jitterTolerence), e.gl.disable(e.gl.BLEND), i.render(e);
            var a = this._positionPass;
            this._framebuffer.attach(this._positionTex), a.setUniform("globalSpeedTex", this._globalSpeedTex), a.setUniform("positionTex", this._positionPrevTex), a.setUniform("forceTex", this._forceTex), a.setUniform("forcePrevTex", this._forcePrevTex), a.render(e), this._framebuffer.unbind(e), this._swapTexture()
        }, n.prototype.update = function(e, t, r) {
            null == t && (t = 1), t = Math.max(t, 1);
            for (var n = 0; n < t; n++) this.step(e);
            r && r()
        }, n.prototype.getNodePositionTexture = function() { return this._inited ? this._positionPrevTex : this._positionSourceTex }, n.prototype.getNodeUV = function(e, t) {
            t = t || [];
            var r = this._positionTex.width,
                n = this._positionTex.height;
            return t[0] = e % r / (r - 1), t[1] = Math.floor(e / r) / (n - 1) || 0, t
        }, n.prototype.getNodePosition = function(e, t) {
            var r = this._positionArr,
                n = this._positionTex.width,
                i = this._positionTex.height,
                a = n * i;
            r && r.length === 4 * a || (r = this._positionArr = new Float32Array(4 * a)), this._framebuffer.bind(e), this._framebuffer.attach(this._positionPrevTex), e.gl.readPixels(0, 0, n, i, e.gl.RGBA, e.gl.FLOAT, r), this._framebuffer.unbind(e), t || (t = new Float32Array(2 * this._nodes.length));
            for (var o = 0; o < this._nodes.length; o++) t[2 * o] = r[4 * o], t[2 * o + 1] = r[4 * o + 1];
            return t
        }, n.prototype.getTextureData = function(e, t) {
            var r = this["_" + t + "Tex"],
                n = r.width,
                i = r.height;
            this._framebuffer.bind(e), this._framebuffer.attach(r);
            var a = new Float32Array(n * i * 4);
            return e.gl.readPixels(0, 0, n, i, e.gl.RGBA, e.gl.FLOAT, a), this._framebuffer.unbind(e), a
        }, n.prototype.getTextureSize = function() { return { width: this._positionTex.width, height: this._positionTex.height } }, n.prototype.isFinished = function(e, t) { var r = this.getTextureData(e, "globalSpeed"); return this._inited && r[0] < t && this._frame > 10 }, n.prototype._swapTexture = function() {
            var e = this._positionPrevTex;
            this._positionPrevTex = this._positionTex, this._positionTex = e;
            var e = this._forcePrevTex;
            this._forcePrevTex = this._forceTex, this._forceTex = e;
            var e = this._globalSpeedPrevTex;
            this._globalSpeedPrevTex = this._globalSpeedTex, this._globalSpeedTex = e
        }, n.prototype._initFromSource = function(e) { this._framebuffer.attach(this._positionPrevTex), this._framebuffer.bind(e), this._copyPass.setUniform("texture", this._positionSourceTex), this._copyPass.render(e), e.gl.clearColor(0, 0, 0, 0), this._framebuffer.attach(this._forcePrevTex), e.gl.clear(e.gl.COLOR_BUFFER_BIT), this._framebuffer.attach(this._globalSpeedPrevTex), e.gl.clear(e.gl.COLOR_BUFFER_BIT), this._framebuffer.unbind(e) }, n.prototype._resize = function(e, t) {
            ["_positionSourceTex", "_positionTex", "_positionPrevTex", "_forceTex", "_forcePrevTex"].forEach(function(r) { this[r].width = e, this[r].height = t, this[r].dirty() }, this)
        }, n.prototype.dispose = function(e) {
            var t = e.gl;
            this._framebuffer.dispose(t), this._copyPass.dispose(t), this._nodeRepulsionPass.dispose(t), this._positionPass.disable(t), this._globalSpeedPass.dispose(t), this._edgeForceMesh.material.shader.dispose(t), this._edgeForceMesh.geometry.dispose(t), this._weightedSumMesh.material.shader.dispose(t), this._weightedSumMesh.geometry.dispose(t), this._positionSourceTex.dispose(t), this._positionTex.dispose(t), this._positionPrevTex.dispose(t), this._forceTex.dispose(t), this._forcePrevTex.dispose(t), this._weightedSumTex.dispose(t), this._globalSpeedTex.disable(t), this._globalSpeedPrevTex.disable(t)
        }, i.ForceAtlas2GPU = n, e.exports = n
    }, function(e, t, r) {
        var n = r(0),
            i = r(115),
            a = r(24),
            o = n.extendSeriesModel({
                type: "series.graphGL",
                visualColorAccessPath: "itemStyle.color",
                init: function(e) { o.superApply(this, "init", arguments), this.legendDataProvider = function() { return this._categoriesData }, this._updateCategoriesData() },
                mergeOption: function(e) { o.superApply(this, "mergeOption", arguments), this._updateCategoriesData() },
                getFormattedLabel: function(e, t, r, n) {
                    var i = a.getFormattedLabel(this, e, t, r, n);
                    if (null == i) {
                        var o = this.getData(),
                            s = o.dimensions[o.dimensions.length - 1];
                        i = o.get(s, e)
                    }
                    return i
                },
                getInitialData: function(e, t) {
                    function r(e, r) {
                        function i(e) { return e = this.parsePath(e), e && "label" === e[0] ? o : this.parentModel }
                        e.wrapMethod("getItemModel", function(e) {
                            var t = s._categoriesModels,
                                r = e.getShallow("category"),
                                n = t[r];
                            return n && (n.parentModel = e.parentModel, e.parentModel = n), e
                        });
                        var a = s.getModel("edgeLabel"),
                            o = new n.Model({ label: a.option }, a.parentModel, t);
                        r.wrapMethod("getItemModel", function(e) { return e.customizeGetParent(i), e })
                    }
                    var a = e.edges || e.links || [],
                        o = e.data || e.nodes || [],
                        s = this;
                    if (o && a) return i(o, a, this, !0, r).data
                },
                getGraph: function() { return this.getData().graph },
                getEdgeData: function() { return this.getGraph().edgeData },
                getCategoriesData: function() { return this._categoriesData },
                formatTooltip: function(e, t, r) {
                    if ("edge" === r) {
                        var i = this.getData(),
                            a = this.getDataParams(e, r),
                            s = i.graph.getEdgeByIndex(e),
                            u = i.getName(s.node1.dataIndex),
                            h = i.getName(s.node2.dataIndex),
                            l = [];
                        return null != u && l.push(u), null != h && l.push(h), l = n.format.encodeHTML(l.join(" > ")), a.value && (l += " : " + n.format.encodeHTML(a.value)), l
                    }
                    return o.superApply(this, "formatTooltip", arguments)
                },
                _updateCategoriesData: function() {
                    var e = (this.option.categories || []).map(function(e) { return null != e.value ? e : n.util.extend({ value: 0 }, e) }),
                        t = new n.List(["value"], this);
                    t.initData(e), this._categoriesData = t, this._categoriesModels = t.mapArray(function(e) { return t.getItemModel(e, !0) })
                },
                setView: function(e) { null != e.zoom && (this.option.zoom = e.zoom), null != e.offset && (this.option.offset = e.offset) },
                setNodePosition: function(e) {
                    for (var t = 0; t < e.length / 2; t++) {
                        var r = e[2 * t],
                            n = e[2 * t + 1],
                            i = this.getData().getRawDataItem(t);
                        i.x = r, i.y = n
                    }
                },
                isAnimationEnabled: function() { return o.superCall(this, "isAnimationEnabled") && !("force" === this.get("layout") && this.get("force.layoutAnimation")) },
                defaultOption: { zlevel: 10, z: 2, legendHoverLink: !0, layout: "forceAtlas2", forceAtlas2: { initLayout: null, GPU: !0, steps: 1, stopThreshold: 1, repulsionByDegree: !0, linLogMode: !1, strongGravityMode: !1, gravity: 1, edgeWeightInfluence: 1, edgeWeight: [1, 4], nodeWeight: [1, 4], preventOverlap: !1, gravityCenter: null }, focusNodeAdjacency: !0, focusNodeAdjacencyOn: "mouseover", left: "center", top: "center", symbol: "circle", symbolSize: 5, roam: !1, center: null, zoom: 1, label: { show: !1, formatter: "{b}", position: "right", distance: 5, textStyle: { fontSize: 14 } }, itemStyle: {}, lineStyle: { color: "#aaa", width: 1, opacity: .5 }, emphasis: { label: { show: !0 } }, animation: !1 }
            });
        e.exports = o
    }, function(e, t, r) {
        var n = r(0),
            i = r(42),
            a = r(2),
            o = r(21),
            s = r(174),
            u = r(4),
            h = r(112),
            l = r(111),
            c = r(83),
            d = r(1).vec2,
            f = r(170),
            p = r(48);
        a.Shader.import(r(184));
        var _ = 1;
        n.extendChartView({
            type: "graphGL",
            __ecgl__: !0,
            init: function(e, t) { this.groupGL = new a.Node, this.viewGL = new o("orthographic"), this.viewGL.camera.left = this.viewGL.camera.right = 0, this.viewGL.add(this.groupGL), this._pointsBuilder = new p(!0, t), this._forceEdgesMesh = new a.Mesh({ material: new a.Material({ shader: a.createShader("ecgl.forceAtlas2.edges"), transparent: !0, depthMask: !1, depthTest: !1 }), geometry: new a.Geometry({ attributes: { node: new a.Geometry.Attribute("node", "float", 2), color: new a.Geometry.Attribute("color", "float", 4, "COLOR") }, dynamic: !0, mainAttribute: "node" }), renderOrder: -1, mode: a.Mesh.LINES }), this._edgesMesh = new a.Mesh({ material: new a.Material({ shader: a.createShader("ecgl.meshLines2D"), transparent: !0, depthMask: !1, depthTest: !1 }), geometry: new s({ useNativeLine: !1, dynamic: !0 }), culling: !1 }), this._layoutId = 0, this._control = new f({ zr: t.getZr(), viewGL: this.viewGL }), this._control.setTarget(this.groupGL), this._control.init(), this._clickHandler = this._clickHandler.bind(this) },
            render: function(e, t, r) {
                this.groupGL.add(this._pointsBuilder.rootNode), this._model = e, this._api = r, this._initLayout(e, t, r), this._pointsBuilder.update(e, t, r), this._forceLayoutInstance instanceof h || this.groupGL.remove(this._forceEdgesMesh), this._updateCamera(e, r), this._control.off("update"), this._control.on("update", function() { r.dispatchAction({ type: "graphGLRoam", seriesId: e.id, zoom: this._control.getZoom(), offset: this._control.getOffset() }), this._pointsBuilder.updateView(this.viewGL.camera) }, this), this._control.setZoom(u.firstNotNull(e.get("zoom"), 1)), this._control.setOffset(e.get("offset") || [0, 0]);
                var n = this._pointsBuilder.getPointsMesh();
                if (n.off("mousemove", this._mousemoveHandler), n.off("mouseout", this._mouseOutHandler, this), r.getZr().off("click", this._clickHandler), this._pointsBuilder.highlightOnMouseover = !0, e.get("focusNodeAdjacency")) { var i = e.get("focusNodeAdjacencyOn"); "click" === i ? r.getZr().on("click", this._clickHandler) : "mouseover" === i && (n.on("mousemove", this._mousemoveHandler, this), n.on("mouseout", this._mouseOutHandler, this), this._pointsBuilder.highlightOnMouseover = !1) }
                this._lastMouseOverDataIndex = -1
            },
            _clickHandler: function(e) {
                if (!this._layouting) {
                    var t = this._pointsBuilder.getPointsMesh().dataIndex;
                    t >= 0 ? this._api.dispatchAction({ type: "graphGLFocusNodeAdjacency", seriesId: this._model.id, dataIndex: t }) : this._api.dispatchAction({ type: "graphGLUnfocusNodeAdjacency", seriesId: this._model.id })
                }
            },
            _mousemoveHandler: function(e) {
                if (!this._layouting) {
                    var t = this._pointsBuilder.getPointsMesh().dataIndex;
                    t >= 0 ? t !== this._lastMouseOverDataIndex && this._api.dispatchAction({ type: "graphGLFocusNodeAdjacency", seriesId: this._model.id, dataIndex: t }) : this._mouseOutHandler(e), this._lastMouseOverDataIndex = t
                }
            },
            _mouseOutHandler: function(e) { this._layouting || (this._api.dispatchAction({ type: "graphGLUnfocusNodeAdjacency", seriesId: this._model.id }), this._lastMouseOverDataIndex = -1) },
            _updateForceEdgesGeometry: function(e, t) {
                var r = this._forceEdgesMesh.geometry,
                    n = t.getEdgeData(),
                    i = 0,
                    o = this._forceLayoutInstance,
                    s = 2 * n.count();
                r.attributes.node.init(s), r.attributes.color.init(s), n.each(function(t) {
                    var s = e[t];
                    r.attributes.node.set(i, o.getNodeUV(s.node1)), r.attributes.node.set(i + 1, o.getNodeUV(s.node2));
                    var h = n.getItemVisual(s.dataIndex, "color"),
                        l = a.parseColor(h);
                    l[3] *= u.firstNotNull(n.getItemVisual(s.dataIndex, "opacity"), 1), r.attributes.color.set(i, l), r.attributes.color.set(i + 1, l), i += 2
                }), r.dirty()
            },
            _updateMeshLinesGeometry: function() {
                var e = this._model.getEdgeData(),
                    t = this._edgesMesh.geometry,
                    e = this._model.getEdgeData(),
                    r = this._model.getData().getLayout("points");
                t.resetOffset(), t.setVertexCount(e.count() * t.getLineVertexCount()), t.setTriangleCount(e.count() * t.getLineTriangleCount());
                var n = [],
                    i = [],
                    o = ["lineStyle", "width"];
                this._originalEdgeColors = new Float32Array(4 * e.count()), this._edgeIndicesMap = new Float32Array(e.count()), e.each(function(s) {
                    var h = e.graph.getEdgeByIndex(s),
                        l = 2 * h.node1.dataIndex,
                        c = 2 * h.node2.dataIndex;
                    n[0] = r[l], n[1] = r[l + 1], i[0] = r[c], i[1] = r[c + 1];
                    var d = e.getItemVisual(h.dataIndex, "color"),
                        f = a.parseColor(d);
                    f[3] *= u.firstNotNull(e.getItemVisual(h.dataIndex, "opacity"), 1);
                    var p = e.getItemModel(h.dataIndex),
                        _ = u.firstNotNull(p.get(o), 1) * this._api.getDevicePixelRatio();
                    t.addLine(n, i, f, _);
                    for (var m = 0; m < 4; m++) this._originalEdgeColors[4 * h.dataIndex + m] = f[m];
                    this._edgeIndicesMap[h.dataIndex] = s
                }, !1, this), t.dirty()
            },
            _updateForceNodesGeometry: function(e) {
                for (var t = this._pointsBuilder.getPointsMesh(), r = [], n = 0; n < e.count(); n++) this._forceLayoutInstance.getNodeUV(n, r), t.geometry.attributes.position.set(n, r);
                t.geometry.dirty("position")
            },
            _initLayout: function(e, t, r) {
                var a = e.get("layout"),
                    o = e.getGraph(),
                    s = e.getBoxLayoutParams(),
                    c = i.getLayoutRect(s, { width: r.getWidth(), height: r.getHeight() });
                "force" === a && (a = "forceAtlas2"), this.stopLayout(e, t, r, { beforeLayout: !0 });
                var d = e.getData(),
                    f = e.getData();
                if ("forceAtlas2" === a) {
                    var p = e.getModel("forceAtlas2"),
                        _ = this._forceLayoutInstance,
                        m = [],
                        g = [],
                        v = d.getDataExtent("value"),
                        y = f.getDataExtent("value"),
                        x = u.firstNotNull(p.get("edgeWeight"), 1),
                        T = u.firstNotNull(p.get("nodeWeight"), 1);
                    "number" == typeof x && (x = [x, x]), "number" == typeof T && (T = [T, T]);
                    var b = 0,
                        w = {},
                        E = new Float32Array(2 * d.count());
                    if (o.eachNode(function(e) {
                            var t, r, i = e.dataIndex,
                                a = d.get("value", i);
                            if (d.hasItemOption) {
                                var o = d.getItemModel(i);
                                t = o.get("x"), r = o.get("y")
                            }
                            null == t && (t = c.x + Math.random() * c.width, r = c.y + Math.random() * c.height), E[2 * b] = t, E[2 * b + 1] = r, w[e.id] = b++;
                            var s = n.number.linearMap(a, v, T);
                            isNaN(s) && (s = isNaN(T[0]) ? 1 : T[0]), m.push({ x: t, y: r, mass: s, size: d.getItemVisual(i, "symbolSize") })
                        }), d.setLayout("points", E), o.eachEdge(function(e) {
                            var t = e.dataIndex,
                                r = d.get("value", t),
                                i = n.number.linearMap(r, y, x);
                            isNaN(i) && (i = isNaN(x[0]) ? 1 : x[0]), g.push({ node1: w[e.node1.id], node2: w[e.node2.id], weight: i, dataIndex: t })
                        }), !_) {
                        var S = p.get("GPU");
                        this._forceLayoutInstance && ((!S || this._forceLayoutInstance instanceof h) && (S || this._forceLayoutInstance instanceof l) || (this._forceLayoutInstanceToDispose = this._forceLayoutInstance)), _ = this._forceLayoutInstance = S ? new h : new l
                    }
                    _.initData(m, g), _.updateOption(p.option), this._updateForceEdgesGeometry(_.getEdges(), e), this._updatePositionTexture(), r.dispatchAction({ type: "graphGLStartLayout" })
                } else {
                    var E = new Float32Array(2 * d.count()),
                        b = 0;
                    o.eachNode(function(e) {
                        var t, r, n = e.dataIndex;
                        if (d.hasItemOption) {
                            var i = d.getItemModel(n);
                            t = i.get("x"), r = i.get("y")
                        }
                        E[b++] = t, E[b++] = r
                    }), d.setLayout("points", E), this._updateAfterLayout(e, t, r)
                }
            },
            _updatePositionTexture: function() {
                var e = this._forceLayoutInstance.getNodePositionTexture();
                this._pointsBuilder.setPositionTexture(e), this._forceEdgesMesh.material.set("positionTex", e)
            },
            startLayout: function(e, t, r, n) {
                var i = this.viewGL,
                    r = this._api,
                    a = this._forceLayoutInstance,
                    o = this._model.getData(),
                    s = this._model.getModel("forceAtlas2");
                if (a && (this.groupGL.remove(this._edgesMesh), this.groupGL.add(this._forceEdgesMesh), this._forceLayoutInstance)) {
                    this._updateForceNodesGeometry(e.getData()), this._pointsBuilder.hideLabels();
                    var u = this,
                        h = this._layoutId = _++,
                        l = s.getShallow("stopThreshold"),
                        d = s.getShallow("steps"),
                        f = 0,
                        p = Math.max(2 * d, 20),
                        m = function(t) { if (t === u._layoutId) return a.isFinished(i.layer.renderer, l) ? (r.dispatchAction({ type: "graphGLStopLayout" }), void r.dispatchAction({ type: "graphGLFinishLayout", points: o.getLayout("points") })) : void a.update(i.layer.renderer, d, function() { u._updatePositionTexture(), f += d, f >= p && (u._syncNodePosition(e), f = 0), r.getZr().refresh(), c(function() { m(t) }) }) };
                    c(function() { u._forceLayoutInstanceToDispose && (u._forceLayoutInstanceToDispose.dispose(i.layer.renderer), u._forceLayoutInstanceToDispose = null), m(h) }), this._layouting = !0
                }
            },
            stopLayout: function(e, t, r, n) { this._layoutId = 0, this.groupGL.remove(this._forceEdgesMesh), this.groupGL.add(this._edgesMesh), this._forceLayoutInstance && this.viewGL.layer && (n && n.beforeLayout || (this._syncNodePosition(e), this._updateAfterLayout(e, t, r)), this._api.getZr().refresh(), this._layouting = !1) },
            _syncNodePosition: function(e) {
                var t = this._forceLayoutInstance.getNodePosition(this.viewGL.layer.renderer);
                e.getData().setLayout("points", t), e.setNodePosition(t)
            },
            _updateAfterLayout: function(e, t, r) { this._updateMeshLinesGeometry(), this._pointsBuilder.removePositionTexture(), this._pointsBuilder.updateLayout(e, t, r), this._pointsBuilder.updateView(this.viewGL.camera), this._pointsBuilder.updateLabels(), this._pointsBuilder.showLabels() },
            focusNodeAdjacency: function(e, t, r, n) {
                var i = this._model.getData();
                this._downplayAll();
                var a = n.dataIndex,
                    o = i.graph,
                    s = [],
                    u = o.getNodeByIndex(a);
                s.push(u), u.edges.forEach(function(e) { e.dataIndex < 0 || (e.node1 !== u && s.push(e.node1), e.node2 !== u && s.push(e.node2)) }, this), this._pointsBuilder.fadeOutAll(.05), this._fadeOutEdgesAll(.05), s.forEach(function(e) { this._pointsBuilder.highlight(i, e.dataIndex) }, this), this._pointsBuilder.updateLabels(s.map(function(e) { return e.dataIndex }));
                var h = [];
                u.edges.forEach(function(e) { e.dataIndex >= 0 && (this._highlightEdge(e.dataIndex), h.push(e)) }, this), this._focusNodes = s, this._focusEdges = h
            },
            unfocusNodeAdjacency: function(e, t, r, n) { this._downplayAll(), this._pointsBuilder.fadeInAll(), this._fadeInEdgesAll(), this._pointsBuilder.updateLabels() },
            _highlightEdge: function(e) {
                var t = this._model.getEdgeData().getItemModel(e),
                    r = a.parseColor(t.get("emphasis.lineStyle.color") || t.get("lineStyle.color")),
                    n = u.firstNotNull(t.get("emphasis.lineStyle.opacity"), t.get("lineStyle.opacity"), 1);
                r[3] *= n, this._edgesMesh.geometry.setItemColor(this._edgeIndicesMap[e], r)
            },
            _downplayAll: function() { this._focusNodes && this._focusNodes.forEach(function(e) { this._pointsBuilder.downplay(this._model.getData(), e.dataIndex) }, this), this._focusEdges && this._focusEdges.forEach(function(e) { this._downplayEdge(e.dataIndex) }, this) },
            _downplayEdge: function(e) {
                var t = this._getColor(e, []);
                this._edgesMesh.geometry.setItemColor(this._edgeIndicesMap[e], t)
            },
            _setEdgeFade: function() { var e = []; return function(t, r) { this._getColor(t, e), e[3] *= r, this._edgesMesh.geometry.setItemColor(this._edgeIndicesMap[t], e) } }(),
            _getColor: function(e, t) { for (var r = 0; r < 4; r++) t[r] = this._originalEdgeColors[4 * e + r]; return t },
            _fadeOutEdgesAll: function(e) { this._model.getData().graph.eachEdge(function(t) { this._setEdgeFade(t.dataIndex, e) }, this) },
            _fadeInEdgesAll: function() { this._fadeOutEdgesAll(1) },
            _updateCamera: function(e, t) {
                this.viewGL.setViewport(0, 0, t.getWidth(), t.getHeight(), t.getDevicePixelRatio());
                for (var r = this.viewGL.camera, n = e.getData(), i = n.getLayout("points"), a = d.create(1 / 0, 1 / 0), o = d.create(-1 / 0, -1 / 0), s = [], u = 0; u < i.length;) s[0] = i[u++], s[1] = i[u++], d.min(a, a, s), d.max(o, o, s);
                var h = (o[1] + a[1]) / 2,
                    l = (o[0] + a[0]) / 2;
                if (!(l > r.left && l < r.right && h < r.bottom && h > r.top)) {
                    var c = Math.max(o[0] - a[0], 10),
                        f = c / t.getWidth() * t.getHeight();
                    c *= 1.4, f *= 1.4, a[0] -= .2 * c, r.left = a[0], r.top = h - f / 2, r.bottom = h + f / 2, r.right = c + a[0], r.near = 0, r.far = 100
                }
            },
            dispose: function() {
                var e = this.viewGL.layer.renderer;
                this._forceLayoutInstance && this._forceLayoutInstance.dispose(e), this.groupGL.removeAll(), this.stopLayout()
            },
            remove: function() { this.groupGL.removeAll(), this._control.dispose() }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(193),
            a = r(194),
            o = r(4);
        e.exports = function(e, t, r, s, u) {
            for (var h = new i(s), l = 0; l < e.length; l++) h.addNode(o.firstNotNull(e[l].id, e[l].name, l), l);
            for (var c = [], d = [], f = 0, l = 0; l < t.length; l++) {
                var p = t[l],
                    _ = p.source,
                    m = p.target;
                h.addEdge(_, m, f) && (d.push(p), c.push(o.firstNotNull(p.id, _ + " > " + m)), f++)
            }
            var g, v = n.helper.completeDimensions(["value"], e);
            g = new n.List(v, r), g.initData(e);
            var y = new n.List(["value"], r);
            return y.initData(d, c), u && u(g, y), a({ mainData: g, struct: h, structAttr: "graph", datas: { node: g, edge: y }, datasAttr: { node: "data", edge: "edgeData" } }), h.update(), h
        }
    }, function(e, t) { e.exports = "@export ecgl.forceAtlas2.updateNodeRepulsion\n\n#define NODE_COUNT 0\n\nuniform sampler2D positionTex;\n\nuniform vec2 textureSize;\nuniform float gravity;\nuniform float scaling;\nuniform vec2 gravityCenter;\n\nuniform bool strongGravityMode;\nuniform bool preventOverlap;\n\nvarying vec2 v_Texcoord;\n\nvoid main() {\n\n vec4 n0 = texture2D(positionTex, v_Texcoord);\n\n vec2 force = vec2(0.0);\n for (int i = 0; i < NODE_COUNT; i++) {\n vec2 uv = vec2(\n mod(float(i), textureSize.x) / (textureSize.x - 1.0),\n floor(float(i) / textureSize.x) / (textureSize.y - 1.0)\n );\n vec4 n1 = texture2D(positionTex, uv);\n\n vec2 dir = n0.xy - n1.xy;\n float d2 = dot(dir, dir);\n\n if (d2 > 0.0) {\n float factor = 0.0;\n if (preventOverlap) {\n float d = sqrt(d2);\n d = d - n0.w - n1.w;\n if (d > 0.0) {\n factor = scaling * n0.z * n1.z / (d * d);\n }\n else if (d < 0.0) {\n factor = scaling * 100.0 * n0.z * n1.z;\n }\n }\n else {\n factor = scaling * n0.z * n1.z / d2;\n }\n force += dir * factor;\n }\n }\n\n vec2 dir = gravityCenter - n0.xy;\n float d = 1.0;\n if (!strongGravityMode) {\n d = length(dir);\n }\n\n force += dir * n0.z * gravity / (d + 1.0);\n\n gl_FragColor = vec4(force, 0.0, 1.0);\n}\n@end\n\n@export ecgl.forceAtlas2.updateEdgeAttraction.vertex\n\nattribute vec2 node1;\nattribute vec2 node2;\nattribute float weight;\n\nuniform sampler2D positionTex;\nuniform float edgeWeightInfluence;\nuniform bool preventOverlap;\nuniform bool linLogMode;\n\nuniform vec2 windowSize: WINDOW_SIZE;\n\nvarying vec2 v_Force;\n\nvoid main() {\n\n vec4 n0 = texture2D(positionTex, node1);\n vec4 n1 = texture2D(positionTex, node2);\n\n vec2 dir = n1.xy - n0.xy;\n float d = length(dir);\n float w;\n if (edgeWeightInfluence == 0.0) {\n w = 1.0;\n }\n else if (edgeWeightInfluence == 1.0) {\n w = weight;\n }\n else {\n w = pow(weight, edgeWeightInfluence);\n }\n vec2 offset = vec2(1.0 / windowSize.x, 1.0 / windowSize.y);\n vec2 scale = vec2((windowSize.x - 1.0) / windowSize.x, (windowSize.y - 1.0) / windowSize.y);\n vec2 pos = node1 * scale * 2.0 - 1.0;\n gl_Position = vec4(pos + offset, 0.0, 1.0);\n gl_PointSize = 1.0;\n\n float factor;\n if (preventOverlap) {\n d = d - n1.w - n0.w;\n }\n if (d <= 0.0) {\n v_Force = vec2(0.0);\n return;\n }\n\n if (linLogMode) {\n factor = w * log(d) / d;\n }\n else {\n factor = w;\n }\n v_Force = dir * factor;\n}\n@end\n\n@export ecgl.forceAtlas2.updateEdgeAttraction.fragment\n\nvarying vec2 v_Force;\n\nvoid main() {\n gl_FragColor = vec4(v_Force, 0.0, 0.0);\n}\n@end\n\n@export ecgl.forceAtlas2.calcWeightedSum.vertex\n\nattribute vec2 node;\n\nvarying vec2 v_NodeUv;\n\nvoid main() {\n\n v_NodeUv = node;\n gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n gl_PointSize = 1.0;\n}\n@end\n\n@export ecgl.forceAtlas2.calcWeightedSum.fragment\n\nvarying vec2 v_NodeUv;\n\nuniform sampler2D positionTex;\nuniform sampler2D forceTex;\nuniform sampler2D forcePrevTex;\n\nvoid main() {\n vec2 force = texture2D(forceTex, v_NodeUv).rg;\n vec2 forcePrev = texture2D(forcePrevTex, v_NodeUv).rg;\n\n float mass = texture2D(positionTex, v_NodeUv).z;\n float swing = length(force - forcePrev) * mass;\n float traction = length(force + forcePrev) * 0.5 * mass;\n\n gl_FragColor = vec4(swing, traction, 0.0, 0.0);\n}\n@end\n\n@export ecgl.forceAtlas2.calcGlobalSpeed\n\nuniform sampler2D globalSpeedPrevTex;\nuniform sampler2D weightedSumTex;\nuniform float jitterTolerence;\n\nvoid main() {\n vec2 weightedSum = texture2D(weightedSumTex, vec2(0.5)).xy;\n float prevGlobalSpeed = texture2D(globalSpeedPrevTex, vec2(0.5)).x;\n float globalSpeed = jitterTolerence * jitterTolerence\n * weightedSum.y / weightedSum.x;\n if (prevGlobalSpeed > 0.0) {\n globalSpeed = min(globalSpeed / prevGlobalSpeed, 1.5) * prevGlobalSpeed;\n }\n gl_FragColor = vec4(globalSpeed, 0.0, 0.0, 1.0);\n}\n@end\n\n@export ecgl.forceAtlas2.updatePosition\n\nuniform sampler2D forceTex;\nuniform sampler2D forcePrevTex;\nuniform sampler2D positionTex;\nuniform sampler2D globalSpeedTex;\n\nvarying vec2 v_Texcoord;\n\nvoid main() {\n vec2 force = texture2D(forceTex, v_Texcoord).xy;\n vec2 forcePrev = texture2D(forcePrevTex, v_Texcoord).xy;\n vec4 node = texture2D(positionTex, v_Texcoord);\n\n float globalSpeed = texture2D(globalSpeedTex, vec2(0.5)).r;\n float swing = length(force - forcePrev);\n float speed = 0.1 * globalSpeed / (0.1 + globalSpeed * sqrt(swing));\n\n float df = length(force);\n if (df > 0.0) {\n speed = min(df * speed, 10.0) / df;\n\n gl_FragColor = vec4(node.xy + speed * force, node.zw);\n }\n else {\n gl_FragColor = node;\n }\n}\n@end\n\n@export ecgl.forceAtlas2.edges.vertex\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec2 node;\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n\nuniform sampler2D positionTex;\n\nvoid main()\n{\n gl_Position = worldViewProjection * vec4(\n texture2D(positionTex, node).xy, -10.0, 1.0\n );\n v_Color = a_Color;\n}\n@end\n\n@export ecgl.forceAtlas2.edges.fragment\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\nvarying vec4 v_Color;\nvoid main() {\n gl_FragColor = color * v_Color;\n}\n@end" }, function(e, t) {
        function r() {
            function e() { this.subRegions = [], this.nSubRegions = 0, this.node = null, this.mass = 0, this.centerOfMass = null, this.bbox = new Float32Array(4), this.size = 0 }

            function t() { this.position = new Float32Array(2), this.force = i.create(), this.forcePrev = i.create(), this.mass = 1, this.inDegree = 0, this.outDegree = 0 }

            function r(e, t) { this.source = e, this.target = t, this.weight = 1 }

            function n() { this.autoSettings = !0, this.barnesHutOptimize = !0, this.barnesHutTheta = 1.5, this.repulsionByDegree = !0, this.linLogMode = !1, this.strongGravityMode = !1, this.gravity = 1, this.scaling = 1, this.edgeWeightInfluence = 1, this.jitterTolerence = .1, this.preventOverlap = !1, this.dissuadeHubs = !1, this.rootRegion = new e, this.rootRegion.centerOfMass = i.create(), this.nodes = [], this.edges = [], this.bbox = new Float32Array(4), this.gravityCenter = null, this._massArr = null, this._swingingArr = null, this._sizeArr = null, this._globalSpeed = 0 }
            var i = {
                    create: function() { return new Float32Array(2) },
                    dist: function(e, t) {
                        var r = t[0] - e[0],
                            n = t[1] - e[1];
                        return Math.sqrt(r * r + n * n)
                    },
                    len: function(e) {
                        var t = e[0],
                            r = e[1];
                        return Math.sqrt(t * t + r * r)
                    },
                    scaleAndAdd: function(e, t, r, n) { return e[0] = t[0] + r[0] * n, e[1] = t[1] + r[1] * n, e },
                    scale: function(e, t, r) { return e[0] = t[0] * r, e[1] = t[1] * r, e },
                    add: function(e, t, r) { return e[0] = t[0] + r[0], e[1] = t[1] + r[1], e },
                    sub: function(e, t, r) { return e[0] = t[0] - r[0], e[1] = t[1] - r[1], e },
                    normalize: function(e, t) {
                        var r = t[0],
                            n = t[1],
                            i = r * r + n * n;
                        return i > 0 && (i = 1 / Math.sqrt(i), e[0] = t[0] * i, e[1] = t[1] * i), e
                    },
                    negate: function(e, t) { return e[0] = -t[0], e[1] = -t[1], e },
                    copy: function(e, t) { return e[0] = t[0], e[1] = t[1], e },
                    set: function(e, t, r) { return e[0] = t, e[1] = r, e }
                },
                a = e.prototype;
            a.beforeUpdate = function() {
                for (var e = 0; e < this.nSubRegions; e++) this.subRegions[e].beforeUpdate();
                this.mass = 0, this.centerOfMass && (this.centerOfMass[0] = 0, this.centerOfMass[1] = 0), this.nSubRegions = 0, this.node = null
            }, a.afterUpdate = function() { this.subRegions.length = this.nSubRegions; for (var e = 0; e < this.nSubRegions; e++) this.subRegions[e].afterUpdate() }, a.addNode = function(e) {
                if (0 === this.nSubRegions) {
                    if (null == this.node) return void(this.node = e);
                    this._addNodeToSubRegion(this.node), this.node = null
                }
                this._addNodeToSubRegion(e), this._updateCenterOfMass(e)
            }, a.findSubRegion = function(e, t) { for (var r = 0; r < this.nSubRegions; r++) { var n = this.subRegions[r]; if (n.contain(e, t)) return n } }, a.contain = function(e, t) { return this.bbox[0] <= e && this.bbox[2] >= e && this.bbox[1] <= t && this.bbox[3] >= t }, a.setBBox = function(e, t, r, n) { this.bbox[0] = e, this.bbox[1] = t, this.bbox[2] = r, this.bbox[3] = n, this.size = (r - e + n - t) / 2 }, a._newSubRegion = function() { var t = this.subRegions[this.nSubRegions]; return t || (t = new e, this.subRegions[this.nSubRegions] = t), this.nSubRegions++, t }, a._addNodeToSubRegion = function(e) {
                var t = this.findSubRegion(e.position[0], e.position[1]),
                    r = this.bbox;
                if (!t) {
                    var n = (r[0] + r[2]) / 2,
                        i = (r[1] + r[3]) / 2,
                        a = (r[2] - r[0]) / 2,
                        o = (r[3] - r[1]) / 2,
                        s = e.position[0] >= n ? 1 : 0,
                        u = e.position[1] >= i ? 1 : 0,
                        t = this._newSubRegion();
                    t.setBBox(s * a + r[0], u * o + r[1], (s + 1) * a + r[0], (u + 1) * o + r[1])
                }
                t.addNode(e)
            }, a._updateCenterOfMass = function(e) {
                null == this.centerOfMass && (this.centerOfMass = new Float32Array(2));
                var t = this.centerOfMass[0] * this.mass,
                    r = this.centerOfMass[1] * this.mass;
                t += e.position[0] * e.mass, r += e.position[1] * e.mass, this.mass += e.mass, this.centerOfMass[0] = t / this.mass, this.centerOfMass[1] = r / this.mass
            };
            var o = n.prototype;
            o.initNodes = function(e, r, n) {
                var i = r.length;
                this.nodes.length = 0;
                for (var a = void 0 !== n, o = 0; o < i; o++) {
                    var s = new t;
                    s.position[0] = e[2 * o], s.position[1] = e[2 * o + 1], s.mass = r[o], a && (s.size = n[o]), this.nodes.push(s)
                }
                this._massArr = r, this._swingingArr = new Float32Array(i), a && (this._sizeArr = n)
            }, o.initEdges = function(e, t) {
                var n = e.length / 2;
                this.edges.length = 0;
                for (var i = 0; i < n; i++) {
                    var a = e[2 * i],
                        o = e[2 * i + 1],
                        s = this.nodes[a],
                        u = this.nodes[o];
                    if (!s || !u) return void console.error("Node not exists, try initNodes before initEdges");
                    s.outDegree++, u.inDegree++;
                    var h = new r(s, u);
                    t && (h.weight = t[i]), this.edges.push(h)
                }
            }, o.updateSettings = function() {
                if (this.repulsionByDegree)
                    for (var e = 0; e < this.nodes.length; e++) {
                        var t = this.nodes[e];
                        t.mass = t.inDegree + t.outDegree + 1
                    } else
                        for (var e = 0; e < this.nodes.length; e++) {
                            var t = this.nodes[e];
                            t.mass = this._massArr[e]
                        }
            }, o.update = function() {
                var e = this.nodes.length;
                if (this.updateSettings(), this.updateBBox(), this.barnesHutOptimize) {
                    this.rootRegion.setBBox(this.bbox[0], this.bbox[1], this.bbox[2], this.bbox[3]), this.rootRegion.beforeUpdate();
                    for (var t = 0; t < e; t++) this.rootRegion.addNode(this.nodes[t]);
                    this.rootRegion.afterUpdate()
                }
                for (var t = 0; t < e; t++) {
                    var r = this.nodes[t];
                    i.copy(r.forcePrev, r.force), i.set(r.force, 0, 0)
                }
                for (var t = 0; t < e; t++) {
                    var n = this.nodes[t];
                    if (this.barnesHutOptimize) this.applyRegionToNodeRepulsion(this.rootRegion, n);
                    else
                        for (var a = t + 1; a < e; a++) {
                            var o = this.nodes[a];
                            this.applyNodeToNodeRepulsion(n, o, !1)
                        }
                    this.gravity > 0 && (this.strongGravityMode ? this.applyNodeStrongGravity(n) : this.applyNodeGravity(n))
                }
                for (var t = 0; t < this.edges.length; t++) this.applyEdgeAttraction(this.edges[t]);
                for (var s = 0, u = 0, h = i.create(), t = 0; t < e; t++) {
                    var r = this.nodes[t],
                        l = i.dist(r.force, r.forcePrev);
                    s += l * r.mass, i.add(h, r.force, r.forcePrev);
                    u += .5 * i.len(h) * r.mass, this._swingingArr[t] = l
                }
                var c = this.jitterTolerence * this.jitterTolerence * u / s;
                this._globalSpeed > 0 && (c = Math.min(c / this._globalSpeed, 1.5) * this._globalSpeed), this._globalSpeed = c;
                for (var t = 0; t < e; t++) {
                    var r = this.nodes[t],
                        l = this._swingingArr[t],
                        d = .1 * c / (1 + c * Math.sqrt(l)),
                        f = i.len(r.force);
                    f > 0 && (d = Math.min(f * d, 10) / f, i.scaleAndAdd(r.position, r.position, r.force, d))
                }
            }, o.applyRegionToNodeRepulsion = function() {
                var e = i.create();
                return function(t, r) {
                    if (t.node) this.applyNodeToNodeRepulsion(t.node, r, !0);
                    else {
                        i.sub(e, r.position, t.centerOfMass);
                        var n = e[0] * e[0] + e[1] * e[1];
                        if (n > this.barnesHutTheta * t.size * t.size) {
                            var a = this.scaling * r.mass * t.mass / n;
                            i.scaleAndAdd(r.force, r.force, e, a)
                        } else
                            for (var o = 0; o < t.nSubRegions; o++) this.applyRegionToNodeRepulsion(t.subRegions[o], r)
                    }
                }
            }(), o.applyNodeToNodeRepulsion = function() {
                var e = i.create();
                return function(t, r, n) {
                    if (t != r) {
                        i.sub(e, t.position, r.position);
                        var a = e[0] * e[0] + e[1] * e[1];
                        if (0 !== a) {
                            var o;
                            if (this.preventOverlap) {
                                var s = Math.sqrt(a);
                                if ((s = s - t.size - r.size) > 0) o = this.scaling * t.mass * r.mass / (s * s);
                                else {
                                    if (!(s < 0)) return;
                                    o = 100 * this.scaling * t.mass * r.mass
                                }
                            } else o = this.scaling * t.mass * r.mass / a;
                            i.scaleAndAdd(t.force, t.force, e, o), i.scaleAndAdd(r.force, r.force, e, -o)
                        }
                    }
                }
            }(), o.applyEdgeAttraction = function() {
                var e = i.create();
                return function(t) {
                    var r = t.source,
                        n = t.target;
                    i.sub(e, r.position, n.position);
                    var a, o = i.len(e);
                    a = 0 === this.edgeWeightInfluence ? 1 : 1 === this.edgeWeightInfluence ? t.weight : Math.pow(t.weight, this.edgeWeightInfluence);
                    var s;
                    this.preventOverlap && (o = o - r.size - n.size) <= 0 || (s = this.linLogMode ? -a * Math.log(o + 1) / (o + 1) : -a, i.scaleAndAdd(r.force, r.force, e, s), i.scaleAndAdd(n.force, n.force, e, -s))
                }
            }(), o.applyNodeGravity = function() {
                var e = i.create();
                return function(t) {
                    i.sub(e, this.gravityCenter, t.position);
                    var r = i.len(e);
                    i.scaleAndAdd(t.force, t.force, e, this.gravity * t.mass / (r + 1))
                }
            }(), o.applyNodeStrongGravity = function() { var e = i.create(); return function(t) { i.sub(e, this.gravityCenter, t.position), i.scaleAndAdd(t.force, t.force, e, this.gravity * t.mass) } }(), o.updateBBox = function() {
                for (var e = 1 / 0, t = 1 / 0, r = -1 / 0, n = -1 / 0, i = 0; i < this.nodes.length; i++) {
                    var a = this.nodes[i].position;
                    e = Math.min(e, a[0]), t = Math.min(t, a[1]), r = Math.max(r, a[0]), n = Math.max(n, a[1])
                }
                this.bbox[0] = e, this.bbox[1] = t, this.bbox[2] = r, this.bbox[3] = n
            }, o.getGlobalSpeed = function() { return this._globalSpeed };
            var s = null;
            self.onmessage = function(e) {
                switch (e.data.cmd) {
                    case "init":
                        s = new n, s.initNodes(e.data.nodesPosition, e.data.nodesMass, e.data.nodesSize), s.initEdges(e.data.edges, e.data.edgesWeight);
                        break;
                    case "updateConfig":
                        if (s)
                            for (var t in e.data.config) s[t] = e.data.config[t];
                        break;
                    case "update":
                        var r = e.data.steps;
                        if (s) {
                            for (var i = 0; i < r; i++) s.update();
                            for (var a = s.nodes.length, o = new Float32Array(2 * a), i = 0; i < a; i++) {
                                var u = s.nodes[i];
                                o[2 * i] = u.position[0], o[2 * i + 1] = u.position[1]
                            }
                            self.postMessage({ buffer: o.buffer, globalSpeed: s.getGlobalSpeed() }, [o.buffer])
                        } else {
                            var h = new Float32Array;
                            self.postMessage({ buffer: h.buffer, globalSpeed: s.getGlobalSpeed() }, [h.buffer])
                        }
                }
            }
        }
        e.exports = r
    }, function(e, t, r) {
        var n = r(0),
            i = r(29),
            a = n.extendSeriesModel({
                type: "series.line3D",
                dependencies: ["grid3D"],
                visualColorAccessPath: "lineStyle.color",
                getInitialData: function(e, t) {
                    var r = n.helper.completeDimensions(["x", "y", "z"], e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") }),
                        i = new n.List(r, this);
                    return i.initData(e.data), i
                },
                formatTooltip: function(e) { return i(this, e) },
                defaultOption: { coordinateSystem: "cartesian3D", zlevel: -10, grid3DIndex: 0, lineStyle: { width: 2 }, animationDurationUpdate: 500 }
            });
        e.exports = a
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(4),
            o = r(22),
            s = r(9),
            u = r(3),
            h = r(1).vec3,
            l = r(239);
        i.Shader.import(r(41)), e.exports = n.extendChartView({
            type: "line3D",
            __ecgl__: !0,
            init: function(e, t) { this.groupGL = new i.Node, this._api = t },
            render: function(e, t, r) {
                var n = this._prevLine3DMesh;
                this._prevLine3DMesh = this._line3DMesh, this._line3DMesh = n, this._line3DMesh || (this._line3DMesh = new i.Mesh({ geometry: new o({ useNativeLine: !1, sortTriangles: !0 }), material: new i.Material({ shader: i.createShader("ecgl.meshLines3D") }), renderOrder: 10 }), this._line3DMesh.geometry.pick = this._pick.bind(this)), this.groupGL.remove(this._prevLine3DMesh), this.groupGL.add(this._line3DMesh);
                var a = e.coordinateSystem;
                if (a && a.viewGL) {
                    a.viewGL.add(this.groupGL);
                    var s = a.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._line3DMesh.material.shader[s]("fragment", "SRGB_DECODE")
                }
                this._doRender(e, r), this._data = e.getData(), this._camera = a.viewGL.camera, this.updateCamera(), this._updateAnimation(e)
            },
            updateCamera: function() { this._updateNDCPosition() },
            _doRender: function(e, t) {
                var r = e.getData(),
                    n = this._line3DMesh;
                n.geometry.resetOffset();
                var o = r.getLayout("points"),
                    s = [],
                    u = new Float32Array(o.length / 3 * 4),
                    h = 0,
                    l = !1;
                r.each(function(e) {
                    var t = r.getItemVisual(e, "color"),
                        n = r.getItemVisual(e, "opacity");
                    null == n && (n = 1), i.parseColor(t, s), s[3] *= n, u[h++] = s[0], u[h++] = s[1], u[h++] = s[2], u[h++] = s[3], s[3] < .99 && (l = !0)
                }), n.geometry.setVertexCount(n.geometry.getPolylineVertexCount(o)), n.geometry.setTriangleCount(n.geometry.getPolylineTriangleCount(o)), n.geometry.addPolyline(o, u, a.firstNotNull(e.get("lineStyle.width"), 1), !0), n.geometry.dirty(), n.geometry.updateBoundingBox();
                var c = n.material;
                c.transparent = l, c.depthMask = !l;
                var d = e.getModel("debug.wireframe");
                d.get("show") ? (n.geometry.createAttribute("barycentric", "float", 3), n.geometry.generateBarycentric(), n.material.shader.define("both", "WIREFRAME_TRIANGLE"), n.material.set("wireframeLineColor", i.parseColor(d.get("lineStyle.color") || "rgba(0,0,0,0.5)")), n.material.set("wireframeLineWidth", a.firstNotNull(d.get("lineStyle.width"), 1))) : n.material.shader.undefine("both", "WIREFRAME_TRIANGLE"), this._points = o, this._initHandler(e, t)
            },
            _updateAnimation: function(e) {
                i.updateVertexAnimation([
                    ["prevPosition", "position"],
                    ["prevPositionPrev", "positionPrev"],
                    ["prevPositionNext", "positionNext"]
                ], this._prevLine3DMesh, this._line3DMesh, e)
            },
            _initHandler: function(e, t) {
                var r = e.getData(),
                    n = e.coordinateSystem,
                    i = this._line3DMesh,
                    a = -1;
                i.seriesIndex = e.seriesIndex, i.off("mousemove"), i.off("mouseout"), i.on("mousemove", function(e) {
                    var o = n.pointToData(e.point._array),
                        s = r.indicesOfNearest("x", o[0])[0];
                    s !== a && (t.dispatchAction({ type: "grid3DShowAxisPointer", value: [r.get("x", s), r.get("y", s), r.get("z", s)] }), i.dataIndex = s), a = s
                }, this), i.on("mouseout", function(e) { a = -1, i.dataIndex = -1, t.dispatchAction({ type: "grid3DHideAxisPointer" }) }, this)
            },
            _updateNDCPosition: function() {
                var e = new s,
                    t = this._camera;
                s.multiply(e, t.projectionMatrix, t.viewMatrix);
                var r = this._positionNDC,
                    n = this._points,
                    i = n.length / 3;
                r && r.length / 2 === i || (r = this._positionNDC = new Float32Array(2 * i));
                for (var a = [], o = 0; o < i; o++) {
                    var u = 3 * o,
                        l = 2 * o;
                    a[0] = n[u], a[1] = n[u + 1], a[2] = n[u + 2], a[3] = 1, h.transformMat4(a, a, e._array), r[l] = a[0] / a[3], r[l + 1] = a[1] / a[3]
                }
            },
            _pick: function(e, t, r, n, i, a) {
                var o = this._positionNDC,
                    s = this._data.hostModel,
                    h = s.get("lineStyle.width"),
                    c = -1,
                    d = r.viewport.width,
                    f = r.viewport.height,
                    p = .5 * d,
                    _ = .5 * f;
                e = (e + 1) * p, t = (t + 1) * _;
                for (var m = 1; m < o.length / 2; m++) {
                    var g = (o[2 * (m - 1)] + 1) * p,
                        v = (o[2 * (m - 1) + 1] + 1) * _,
                        y = (o[2 * m] + 1) * p,
                        x = (o[2 * m + 1] + 1) * _;
                    if (l.containStroke(g, v, y, x, h, e, t)) { c = (g - e) * (g - e) + (v - t) * (v - t) < (y - e) * (y - e) + (x - t) * (x - t) ? m - 1 : m }
                }
                if (c >= 0) {
                    var T = 3 * c,
                        b = new u(this._points[T], this._points[T + 1], this._points[T + 2]);
                    a.push({ dataIndex: c, point: b, pointWorld: b.clone(), target: this._line3DMesh, distance: this._camera.getWorldPosition().dist(b) })
                }
            },
            remove: function() { this.groupGL.removeAll() },
            dispose: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        var n = r(0);
        n.extendSeriesModel({
            type: "series.lines3D",
            dependencies: ["globe"],
            visualColorAccessPath: "lineStyle.color",
            getInitialData: function(e, t) {
                var r = new n.List(["value"], this);
                return r.hasItemOption = !1, r.initData(e.data, [], function(e, t, n, i) {
                    if (e instanceof Array) return NaN;
                    r.hasItemOption = !0;
                    var a = e.value;
                    return null != a ? a instanceof Array ? a[i] : a : void 0
                }), r
            },
            defaultOption: { coordinateSystem: "globe", globeIndex: 0, geo3DIndex: 0, zlevel: -10, polyline: !1, effect: { show: !1, period: 4, trailWidth: 4, trailLength: .2, spotIntensity: 6 }, silent: !0, blendMode: "source-over", lineStyle: { width: 1, opacity: .5 } }
        })
    }, function(e, t, r) {
        function n(e) { return null != e.radius ? e.radius : null != e.size ? Math.max(e.size[0], e.size[1], e.size[2]) : 100 }
        var i = r(0),
            a = r(2),
            o = r(22),
            s = r(122);
        a.Shader.import(r(41)), e.exports = i.extendChartView({
            type: "lines3D",
            __ecgl__: !0,
            init: function(e, t) { this.groupGL = new a.Node, this._meshLinesMaterial = new a.Material({ shader: a.createShader("ecgl.meshLines3D"), transparent: !0, depthMask: !1 }), this._linesMesh = new a.Mesh({ geometry: new o, material: this._meshLinesMaterial, $ignorePicking: !0 }), this._trailMesh = new s },
            render: function(e, t, r) {
                this.groupGL.add(this._linesMesh);
                var n = e.coordinateSystem,
                    i = e.getData();
                if (n && n.viewGL) {
                    n.viewGL.add(this.groupGL), this._updateLines(e, t, r);
                    var o = n.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._linesMesh.material.shader[o]("fragment", "SRGB_DECODE"), this._trailMesh.material.shader[o]("fragment", "SRGB_DECODE")
                }
                var s = this._trailMesh;
                if (s.stopAnimation(), e.get("effect.show")) {
                    this.groupGL.add(s), s.updateData(i, r, this._linesMesh.geometry), s.__time = s.__time || 0;
                    this._curveEffectsAnimator = s.animate("", { loop: !0 }).when(36e5, { __time: 36e5 }).during(function() { s.setAnimationTime(s.__time) }).start()
                } else this.groupGL.remove(s), this._curveEffectsAnimator = null;
                this._linesMesh.material.blend = this._trailMesh.material.blend = "lighter" === e.get("blendMode") ? a.additiveBlend : null
            },
            pauseEffect: function() { this._curveEffectsAnimator && this._curveEffectsAnimator.pause() },
            resumeEffect: function() { this._curveEffectsAnimator && this._curveEffectsAnimator.resume() },
            toggleEffect: function() {
                var e = this._curveEffectsAnimator;
                e && (e.isPaused() ? e.resume() : e.pause())
            },
            _updateLines: function(e, t, r) {
                var i = e.getData(),
                    o = e.coordinateSystem,
                    s = this._linesMesh.geometry,
                    u = e.get("polyline");
                s.expandLine = !0;
                var h = n(o);
                s.segmentScale = h / 20;
                var l = "lineStyle.width".split("."),
                    c = r.getDevicePixelRatio(),
                    d = 0;
                i.each(function(e) {
                    var t = i.getItemModel(e),
                        r = t.get(l);
                    null == r && (r = 1), i.setItemVisual(e, "lineWidth", r), d = Math.max(r, d)
                }), s.useNativeLine = !1;
                var f = 0,
                    p = 0;
                i.each(function(e) {
                    var t = i.getItemLayout(e);
                    u ? (f += s.getPolylineVertexCount(t), p += s.getPolylineTriangleCount(t)) : (f += s.getCubicCurveVertexCount(t[0], t[1], t[2], t[3]), p += s.getCubicCurveTriangleCount(t[0], t[1], t[2], t[3]))
                }), s.setVertexCount(f), s.setTriangleCount(p), s.resetOffset();
                var _ = [];
                i.each(function(e) {
                    var t = i.getItemLayout(e),
                        r = i.getItemVisual(e, "color"),
                        n = i.getItemVisual(e, "opacity"),
                        o = i.getItemVisual(e, "lineWidth") * c;
                    null == n && (n = 1), _ = a.parseColor(r, _), _[3] *= n, u ? s.addPolyline(t, _, o) : s.addCubicCurve(t[0], t[1], t[2], t[3], _, o)
                }), s.dirty()
            },
            remove: function() { this.groupGL.removeAll() },
            dispose: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        function n(e) { return e > 0 ? 1 : -1 }
        var i = (r(0), r(2)),
            a = r(1).vec3,
            o = r(22);
        i.Shader.import(r(124)), e.exports = i.Mesh.extend(function() {
            var e = new i.Material({ shader: new i.Shader({ vertex: i.Shader.source("ecgl.trail2.vertex"), fragment: i.Shader.source("ecgl.trail2.fragment") }), transparent: !0, depthMask: !1 }),
                t = new o({ dynamic: !0 });
            return t.createAttribute("dist", "float", 1), t.createAttribute("distAll", "float", 1), t.createAttribute("start", "float", 1), { geometry: t, material: e, culling: !1, $ignorePicking: !0 }
        }, {
            updateData: function(e, t, r) {
                var o = e.hostModel,
                    s = this.geometry,
                    u = o.getModel("effect"),
                    h = u.get("trailWidth") * t.getDevicePixelRatio(),
                    l = u.get("trailLength"),
                    c = o.get("effect.constantSpeed"),
                    d = 1e3 * o.get("effect.period"),
                    f = null != c;
                f ? this.material.set("speed", c / 1e3) : this.material.set("period", d), this.material.shader[f ? "define" : "undefine"]("vertex", "CONSTANT_SPEED");
                var p = o.get("polyline");
                s.trailLength = l, this.material.set("trailLength", l), s.resetOffset(), ["position", "positionPrev", "positionNext"].forEach(function(e) { s.attributes[e].value = r.attributes[e].value }), ["dist", "distAll", "start", "offset", "color"].forEach(function(e) { s.attributes[e].init(s.vertexCount) }), s.indices = r.indices;
                var _ = [],
                    m = u.get("trailColor"),
                    g = u.get("trailOpacity"),
                    v = null != m,
                    y = null != g;
                this.updateWorldTransform();
                var x = this.worldTransform.x.len(),
                    T = this.worldTransform.y.len(),
                    b = this.worldTransform.z.len(),
                    w = 0,
                    E = 0;
                e.each(function(t) {
                    var o = e.getItemLayout(t),
                        u = y ? g : e.getItemVisual(t, "opacity"),
                        l = e.getItemVisual(t, "color");
                    null == u && (u = 1), _ = i.parseColor(v ? m : l, _), _[3] *= u;
                    for (var c = p ? r.getPolylineVertexCount(o) : r.getCubicCurveVertexCount(o[0], o[1], o[2], o[3]), S = 0, A = [], M = [], N = w; N < w + c; N++) s.attributes.position.get(N, A), A[0] *= x, A[1] *= T, A[2] *= b, N > w && (S += a.dist(A, M)), s.attributes.dist.set(N, S), a.copy(M, A);
                    E = Math.max(E, S);
                    for (var C = Math.random() * (f ? S : d), N = w; N < w + c; N++) s.attributes.distAll.set(N, S), s.attributes.start.set(N, C), s.attributes.offset.set(N, n(r.attributes.offset.get(N)) * h / 2), s.attributes.color.set(N, _);
                    w += c
                }), this.material.set("spotSize", .1 * E * l), this.material.set("spotIntensity", u.get("spotIntensity")), s.dirty()
            },
            setAnimationTime: function(e) { this.material.set("time", e) }
        })
    }, function(e, t, r) {
        function n(e, t) {
            d.copy(b, e[0]), d.copy(w, e[1]);
            var r = [],
                n = r[0] = g(),
                i = r[1] = g(),
                a = r[2] = g(),
                o = r[3] = g();
            t.dataToPoint(b, n), t.dataToPoint(w, o), f(v, n), _(y, o, n), f(y, y), p(x, y, v), f(x, x), p(y, v, x), m(i, v, y), f(i, i), f(v, o), _(y, n, o), f(y, y), p(x, y, v), f(x, x), p(y, v, x), m(a, v, y), f(a, a), m(T, n, o), f(T, T);
            var s = c.dot(n, T),
                u = c.dot(T, i),
                h = (Math.max(c.len(n), c.len(o)) - s) / u * 2;
            return c.scaleAndAdd(i, n, i, h), c.scaleAndAdd(a, o, a, h), r
        }

        function i(e, t, r) {
            var n = [],
                i = n[0] = c.create(),
                a = n[1] = c.create(),
                o = n[2] = c.create(),
                s = n[3] = c.create();
            t.dataToPoint(e[0], i), t.dataToPoint(e[1], s);
            var u = c.dist(i, s);
            return c.lerp(a, i, s, .3), c.lerp(o, i, s, .3), c.scaleAndAdd(a, a, r, Math.min(.1 * u, 10)), c.scaleAndAdd(o, o, r, Math.min(.1 * u, 10)), n
        }

        function a(e, t) { for (var r = new Float32Array(3 * e.length), n = 0, i = [], a = 0; a < e.length; a++) t.dataToPoint(e[a], i), r[n++] = i[0], r[n++] = i[1], r[n++] = i[2]; return r }

        function o(e) {
            var t = [];
            return e.each(function(r) {
                var n = e.getItemModel(r),
                    i = n.option instanceof Array ? n.option : n.getShallow("coords", !0);
                t.push(i)
            }), { coordsList: t }
        }

        function s(e, t) {
            var r = e.getData(),
                i = e.get("polyline");
            r.setLayout("lineType", i ? "polyline" : "cubicBezier");
            var s = o(r);
            r.each(function(e) {
                var o = s.coordsList[e],
                    u = i ? a : n;
                r.setItemLayout(e, u(o, t))
            })
        }

        function u(e, t, r) {
            var n = e.getData(),
                s = e.get("polyline"),
                u = o(n);
            n.setLayout("lineType", s ? "polyline" : "cubicBezier"), n.each(function(e) {
                var o = u.coordsList[e],
                    h = s ? a(o, t) : i(o, t, r);
                n.setItemLayout(e, h)
            })
        }
        var h = r(0),
            l = r(1),
            c = l.vec3,
            d = l.vec2,
            f = c.normalize,
            p = c.cross,
            _ = c.sub,
            m = c.add,
            g = c.create,
            v = g(),
            y = g(),
            x = g(),
            T = g(),
            b = [],
            w = [];
        h.registerLayout(function(e, t) { e.eachSeriesByType("lines3D", function(e) { var t = e.coordinateSystem; "globe" === t.type ? s(e, t) : "geo3D" === t.type ? u(e, t, [0, 1, 0]) : "mapbox" === t.type && u(e, t, [0, 0, 1]) }) })
    }, function(e, t) { e.exports = "@export ecgl.trail2.vertex\nattribute vec3 position: POSITION;\nattribute vec3 positionPrev;\nattribute vec3 positionNext;\nattribute float offset;\nattribute float dist;\nattribute float distAll;\nattribute float start;\n\nattribute vec4 a_Color : COLOR;\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform vec4 viewport : VIEWPORT;\nuniform float near : NEAR;\n\nuniform float speed : 0;\nuniform float trailLength: 0.3;\nuniform float time;\nuniform float period: 1000;\n\nuniform float spotSize: 1;\n\nvarying vec4 v_Color;\nvarying float v_Percent;\nvarying float v_SpotPercent;\n\n@import ecgl.common.wireframe.vertexHeader\n\n@import ecgl.lines3D.clipNear\n\nvoid main()\n{\n @import ecgl.lines3D.expandLine\n\n gl_Position = currProj;\n\n v_Color = a_Color;\n\n @import ecgl.common.wireframe.vertexMain\n\n#ifdef CONSTANT_SPEED\n float t = mod((speed * time + start) / distAll, 1. + trailLength) - trailLength;\n#else\n float t = mod((time + start) / period, 1. + trailLength) - trailLength;\n#endif\n\n float trailLen = distAll * trailLength;\n\n v_Percent = (dist - t * distAll) / trailLen;\n\n v_SpotPercent = spotSize / distAll;\n\n }\n@end\n\n\n@export ecgl.trail2.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\nuniform float spotIntensity: 5;\n\nvarying vec4 v_Color;\nvarying float v_Percent;\nvarying float v_SpotPercent;\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.util.srgb\n\nvoid main()\n{\n if (v_Percent > 1.0 || v_Percent < 0.0) {\n discard;\n }\n\n float fade = v_Percent;\n\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(color * v_Color);\n#else\n gl_FragColor = color * v_Color;\n#endif\n\n @import ecgl.common.wireframe.fragmentMain\n\n if (v_Percent > (1.0 - v_SpotPercent)) {\n gl_FragColor.rgb *= spotIntensity;\n }\n\n gl_FragColor.a *= fade;\n}\n\n@end" }, function(e, t, r) {
        var n = r(0),
            i = r(38),
            a = r(32),
            o = r(31),
            s = r(33),
            u = r(64),
            h = r(24),
            l = r(29),
            c = n.extendSeriesModel({
                type: "series.map3D",
                layoutMode: "box",
                coordinateSystem: null,
                visualColorAccessPath: "itemStyle.areaColor",
                optionUpdated: function(e) { e = e || {}; var t = this.get("coordinateSystem"); if (null != t && "geo3D" !== t) { this.get("groundPlane.show") && (this.option.groundPlane.show = !1) } },
                getInitialData: function(e) {
                    e.data = this.getFilledRegions(e.data, e.map);
                    var t = n.helper.completeDimensions(["value"], e.data),
                        r = new n.List(t, this);
                    r.initData(e.data);
                    var i = {};
                    return r.each(function(e) {
                        var t = r.getName(e),
                            n = r.getItemModel(e);
                        i[t] = n
                    }), this._regionModelMap = i, r
                },
                formatTooltip: function(e) { return l(this, e) },
                getRegionModel: function(e) { return this._regionModelMap[e] || new n.Model(null, this) },
                getFormattedLabel: function(e, t) { var r = h.getFormattedLabel(this, e, t); return null == r && (r = this.getData().getName(e)), r },
                defaultOption: { coordinateSystem: "geo3D", data: null }
            });
        n.util.merge(c.prototype, u), n.util.merge(c.prototype, i), n.util.merge(c.prototype, a), n.util.merge(c.prototype, o), n.util.merge(c.prototype, s), e.exports = c
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(40),
            o = r(30),
            s = r(61);
        e.exports = n.extendChartView({
            type: "map3D",
            __ecgl__: !0,
            init: function(e, t) { this._geo3DBuilder = new s(t), this.groupGL = new i.Node },
            render: function(e, t, r) {
                var n = e.coordinateSystem;
                if (n && n.viewGL) {
                    this.groupGL.add(this._geo3DBuilder.rootNode), n.viewGL.add(this.groupGL);
                    var i;
                    if ("geo3D" === n.type) {
                        i = n, this._sceneHelper || (this._sceneHelper = new o, this._sceneHelper.initLight(this.groupGL)), this._sceneHelper.setScene(n.viewGL.scene), this._sceneHelper.updateLight(e), n.viewGL.setPostEffect(e.getModel("postEffect"), r), n.viewGL.setTemporalSuperSampling(e.getModel("temporalSuperSampling"));
                        var s = this._control;
                        s || (s = this._control = new a({ zr: r.getZr() }), this._control.init());
                        var u = e.getModel("viewControl");
                        s.setViewGL(n.viewGL), s.setFromViewControlModel(u, 0), s.off("update"), s.on("update", function() { r.dispatchAction({ type: "map3DChangeCamera", alpha: s.getAlpha(), beta: s.getBeta(), distance: s.getDistance(), from: this.uid, map3DId: e.id }) }), this._geo3DBuilder.extrudeY = !0
                    } else this._control && (this._control.dispose(), this._control = null), this._sceneHelper && (this._sceneHelper.dispose(), this._sceneHelper = null), i = e.getData().getLayout("geo3D"), this._geo3DBuilder.extrudeY = !1;
                    this._geo3DBuilder.update(e, i, t, r);
                    var h = n.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._geo3DBuilder.rootNode.traverse(function(e) { e.material && e.material.shader[h]("fragment", "SRGB_DECODE") })
                }
            },
            afterRender: function(e, t, r, n) {
                var i = n.renderer,
                    a = e.coordinateSystem;
                a && "geo3D" === a.type && (this._sceneHelper.updateAmbientCubemap(i, e, r), this._sceneHelper.updateSkybox(i, e, r))
            },
            dispose: function() { this.groupGL.removeAll(), this._control.dispose() }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(24),
            a = r(29);
        n.extendSeriesModel({
            type: "series.scatter3D",
            dependencies: ["globe", "grid3D", "geo3D"],
            visualColorAccessPath: "itemStyle.color",
            getInitialData: function(e, t) {
                var r = n.getCoordinateSystemDimensions(this.get("coordinateSystem")) || ["x", "y", "z"],
                    i = n.helper.completeDimensions(r, e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") }),
                    a = new n.List(i, this);
                return a.initData(e.data), a
            },
            getFormattedLabel: function(e, t, r, n) {
                var a = i.getFormattedLabel(this, e, t, r, n);
                if (null == a) {
                    var o = this.getData(),
                        s = o.dimensions[o.dimensions.length - 1];
                    a = o.get(s, e)
                }
                return a
            },
            formatTooltip: function(e) { return a(this, e) },
            defaultOption: { coordinateSystem: "cartesian3D", zlevel: -10, grid3DIndex: 0, globeIndex: 0, symbol: "circle", symbolSize: 10, blendMode: "source-over", label: { show: !1, position: "right", distance: 5, textStyle: { fontSize: 14, color: "#000", backgroundColor: "rgba(255,255,255,0.7)", padding: 3, borderRadius: 3 } }, itemStyle: { opacity: .8 }, emphasis: { label: { show: !0 } }, animationDurationUpdate: 500 }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(4),
            o = r(24),
            s = r(48);
        n.extendChartView({
            type: "scatter3D",
            __ecgl__: !0,
            init: function(e, t) {
                this.groupGL = new i.Node;
                var r = new s(!1, t);
                this._pointsBuilder = r
            },
            render: function(e, t, r) {
                this.groupGL.add(this._pointsBuilder.rootNode);
                var n = e.coordinateSystem;
                n && n.viewGL && (n.viewGL.add(this.groupGL), this._pointsBuilder.update(e, t, r), this._pointsBuilder.updateView(n.viewGL.camera), this._camera = n.viewGL.camera)
            },
            updateLayout: function(e, t, r) { this._pointsBuilder.updateLayout(e, t, r), this._pointsBuilder.updateView(this._camera) },
            updateCamera: function() { this._pointsBuilder.updateView(this._camera) },
            highlight: function(e, t, r, n) { this._toggleStatus("highlight", e, t, r, n) },
            downplay: function(e, t, r, n) { this._toggleStatus("downplay", e, t, r, n) },
            _toggleStatus: function(e, t, r, i, s) {
                var u = t.getData(),
                    h = a.queryDataIndex(u, s),
                    l = this._pointsBuilder;
                null != h ? n.util.each(o.normalizeToArray(h), function(t) { "highlight" === e ? l.highlight(u, t) : l.downplay(u, t) }, this) : u.each(function(t) { "highlight" === e ? l.highlight(u, t) : l.downplay(u, t) })
            },
            dispose: function() { this.groupGL.removeAll() },
            remove: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        var n = r(0);
        n.extendSeriesModel({ type: "series.scatterGL", dependencies: ["grid", "polar", "geo", "singleAxis"], visualColorAccessPath: "itemStyle.color", getInitialData: function() { return n.helper.createList(this) }, defaultOption: { coordinateSystem: "cartesian2d", zlevel: 10, symbol: "circle", symbolSize: 10, blendMode: "source-over", itemStyle: { opacity: .8 } } })
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(21),
            o = r(48);
        n.extendChartView({
            type: "scatterGL",
            __ecgl__: !0,
            init: function(e, t) { this.groupGL = new i.Node, this.viewGL = new a("orthographic"), this.viewGL.add(this.groupGL), this._pointsBuilder = new o(!0, t) },
            render: function(e, t, r) { this.groupGL.add(this._pointsBuilder.rootNode), this._updateCamera(r.getWidth(), r.getHeight(), r.getDevicePixelRatio()), this._pointsBuilder.update(e, t, r), this._pointsBuilder.updateView(this.viewGL.camera) },
            updateLayout: function(e, t, r) { this._pointsBuilder.updateLayout(e, t, r), this._pointsBuilder.updateView(this.viewGL.camera) },
            _updateCamera: function(e, t, r) {
                this.viewGL.setViewport(0, 0, e, t, r);
                var n = this.viewGL.camera;
                n.left = n.top = 0, n.bottom = t, n.right = e, n.near = 0, n.far = 100
            },
            dispose: function() { this.groupGL.removeAll() },
            remove: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(33),
            a = r(29),
            o = n.extendSeriesModel({
                type: "series.surface",
                dependencies: ["globe", "grid3D", "geo3D"],
                visualColorAccessPath: "itemStyle.color",
                formatTooltip: function(e) { return a(this, e) },
                getInitialData: function(e, t) {
                    function r(e) { return !(isNaN(e.min) || isNaN(e.max) || isNaN(e.step)) }

                    function i(e) { var t = n.number.getPrecisionSafe; return Math.max(t(e.min), t(e.max), t(e.step)) + 1 }
                    var a = e.data;
                    if (!a)
                        if (a = [], e.parametric) {
                            var o = e.parametricEquation || {},
                                s = o.u || {},
                                u = o.v || {};
                            ["u", "v"].forEach(function(e) { r(o[e]) }), ["x", "y", "z"].forEach(function(e) { o[e] });
                            for (var h = i(s), l = i(u), c = u.min; c < u.max + .999 * u.step; c += u.step)
                                for (var d = s.min; d < s.max + .999 * s.step; d += s.step) {
                                    var f = n.number.round(Math.min(d, s.max), h),
                                        p = n.number.round(Math.min(c, u.max), l),
                                        _ = o.x(f, p),
                                        m = o.y(f, p),
                                        g = o.z(f, p);
                                    a.push([_, m, g, f, p])
                                }
                        } else {
                            var v = e.equation || {},
                                y = v.x || {},
                                x = v.y || {};
                            if (["x", "y"].forEach(function(e) { r(v[e]) }), "function" != typeof v.z) return;
                            for (var T = i(y), b = i(x), m = x.min; m < x.max + .999 * x.step; m += x.step)
                                for (var _ = y.min; _ < y.max + .999 * y.step; _ += y.step) {
                                    var w = n.number.round(Math.min(_, y.max), T),
                                        E = n.number.round(Math.min(m, x.max), b),
                                        g = v.z(w, E);
                                    a.push([w, E, g])
                                }
                        }
                    var S = ["x", "y", "z"];
                    e.parametric && S.push("u", "v"), S = n.helper.completeDimensions(S, e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") });
                    var A = new n.List(S, this);
                    return A.initData(a), A
                },
                defaultOption: { coordinateSystem: "cartesian3D", zlevel: -10, grid3DIndex: 0, shading: "lambert", parametric: !1, wireframe: { show: !0, lineStyle: { color: "rgba(0,0,0,0.5)", width: 1 } }, equation: { x: { min: -1, max: 1, step: .1 }, y: { min: -1, max: 1, step: .1 }, z: null }, parametricEquation: { u: { min: -1, max: 1, step: .1 }, v: { min: -1, max: 1, step: .1 }, x: null, y: null, z: null }, itemStyle: {}, animationDurationUpdate: 500 }
            });
        n.util.merge(o.prototype, i), e.exports = o
    }, function(e, t, r) {
        function n(e) { return isNaN(e[0]) || isNaN(e[1]) || isNaN(e[2]) }
        var i = r(0),
            a = r(2),
            o = (r(4), r(1).vec3),
            s = r(50);
        i.extendChartView({
            type: "surface",
            __ecgl__: !0,
            init: function(e, t) {
                this.groupGL = new a.Node;
                var r = {};
                a.COMMON_SHADERS.forEach(function(e) { r[e] = new a.Material({ shader: a.createShader("ecgl." + e) }), r[e].shader.define("both", "VERTEX_COLOR"), r[e].shader.define("fragment", "DOUBLE_SIDED") }), this._materials = r
            },
            render: function(e, t, r) {
                var n = this._prevSurfaceMesh;
                this._prevSurfaceMesh = this._surfaceMesh, this._surfaceMesh = n, this._surfaceMesh || (this._surfaceMesh = this._createSurfaceMesh()), this.groupGL.remove(this._prevSurfaceMesh), this.groupGL.add(this._surfaceMesh);
                var i = e.coordinateSystem,
                    o = e.get("shading"),
                    s = e.getData();
                if (this._materials[o] ? this._surfaceMesh.material = this._materials[o] : this._surfaceMesh.material = this._materials.lambert, a.setMaterialFromModel(o, this._surfaceMesh.material, e, r), i && i.viewGL) {
                    i.viewGL.add(this.groupGL);
                    var u = i.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._surfaceMesh.material.shader[u]("fragment", "SRGB_DECODE")
                }
                var h = e.get("parametric"),
                    l = this._getDataShape(s, h),
                    c = e.getModel("wireframe"),
                    d = c.get("lineStyle.width"),
                    f = c.get("show") && d > 0;
                this._updateSurfaceMesh(this._surfaceMesh, e, l, f);
                var p = this._surfaceMesh.material;
                f ? (p.shader.define("WIREFRAME_QUAD"), p.set("wireframeLineWidth", d), p.set("wireframeLineColor", a.parseColor(c.get("lineStyle.color")))) : p.shader.undefine("WIREFRAME_QUAD"), this._initHandler(e, r), this._updateAnimation(e)
            },
            _updateAnimation: function(e) {
                a.updateVertexAnimation([
                    ["prevPosition", "position"],
                    ["prevNormal", "normal"]
                ], this._prevSurfaceMesh, this._surfaceMesh, e)
            },
            _createSurfaceMesh: function() { var e = new a.Mesh({ geometry: new a.Geometry({ dynamic: !0, sortTriangles: !0 }), shadowDepthMaterial: new a.Material({ shader: new a.Shader({ vertex: a.Shader.source("ecgl.sm.depth.vertex"), fragment: a.Shader.source("ecgl.sm.depth.fragment") }) }), culling: !1, renderOrder: 10, renderNormal: !0 }); return e.geometry.createAttribute("barycentric", "float", 4), e.geometry.createAttribute("prevPosition", "float", 3), e.geometry.createAttribute("prevNormal", "float", 3), i.util.extend(e.geometry, s), e },
            _initHandler: function(e, t) {
                function r(e, t) {
                    for (var r = 1 / 0, n = -1, a = [], s = 0; s < e.length; s++) {
                        i.geometry.attributes.position.get(e[s], a);
                        var u = o.dist(t._array, a);
                        u < r && (r = u, n = e[s])
                    }
                    return n
                }
                var n = e.getData(),
                    i = this._surfaceMesh,
                    a = e.coordinateSystem;
                i.seriesIndex = e.seriesIndex;
                var s = -1;
                i.off("mousemove"), i.off("mouseout"), i.on("mousemove", function(e) {
                    var u = r(e.triangle, e.point);
                    if (u >= 0) {
                        var h = [];
                        i.geometry.attributes.position.get(u, h);
                        for (var l = a.pointToData(h), c = 1 / 0, d = -1, f = [], p = 0; p < n.count(); p++) {
                            f[0] = n.get("x", p), f[1] = n.get("y", p), f[2] = n.get("z", p);
                            var _ = o.squaredDistance(f, l);
                            _ < c && (d = p, c = _)
                        }
                        d !== s && t.dispatchAction({ type: "grid3DShowAxisPointer", value: l }), s = d, i.dataIndex = d
                    } else i.dataIndex = -1
                }, this), i.on("mouseout", function(e) { s = -1, i.dataIndex = -1, t.dispatchAction({ type: "grid3DHideAxisPointer" }) }, this)
            },
            _updateSurfaceMesh: function(e, t, r, i) {
                var s = e.geometry,
                    u = t.getData(),
                    h = u.getLayout("points"),
                    l = 0;
                u.each(function(e) { u.hasValue(e) || l++ });
                var c = l || i,
                    d = s.attributes.position,
                    f = s.attributes.normal,
                    p = s.attributes.texcoord0,
                    _ = s.attributes.barycentric,
                    m = s.attributes.color,
                    g = r.row,
                    v = r.column,
                    y = t.get("shading"),
                    x = "color" !== y;
                if (c) {
                    var T = (g - 1) * (v - 1) * 4;
                    d.init(T), i && _.init(T)
                } else d.value = new Float32Array(h);
                m.init(s.vertexCount), p.init(s.vertexCount);
                var b = [0, 3, 1, 1, 3, 2],
                    w = [
                        [1, 1, 0, 0],
                        [0, 1, 0, 1],
                        [1, 0, 0, 1],
                        [1, 0, 1, 0]
                    ],
                    E = s.indices = new(s.vertexCount > 65535 ? Uint32Array : Uint16Array)((g - 1) * (v - 1) * 6),
                    S = function(e, t, r) { r[1] = e * v + t, r[0] = e * v + t + 1, r[3] = (e + 1) * v + t + 1, r[2] = (e + 1) * v + t },
                    A = !1;
                if (c) {
                    var M = [],
                        N = [],
                        C = 0;
                    x ? f.init(s.vertexCount) : f.value = null;
                    for (var L = [
                            [],
                            [],
                            []
                        ], D = [], I = [], R = o.create(), P = function(e, t, r) { var n = 3 * t; return r[0] = e[n], r[1] = e[n + 1], r[2] = e[n + 2], r }, O = new Float32Array(h.length), F = new Float32Array(h.length / 3 * 4), B = 0; B < u.count(); B++)
                        if (u.hasValue(B)) {
                            var U = a.parseColor(u.getItemVisual(B, "color")),
                                z = u.getItemVisual(B, "opacity");
                            U[3] *= z, U[3] < .99 && (A = !0);
                            for (var G = 0; G < 4; G++) F[4 * B + G] = U[G]
                        }
                    for (var k = [1e7, 1e7, 1e7], B = 0; B < g - 1; B++)
                        for (var H = 0; H < v - 1; H++) {
                            var V = B * (v - 1) + H,
                                W = 4 * V;
                            S(B, H, M);
                            for (var q = !1, G = 0; G < 4; G++) P(h, M[G], N), n(N) && (q = !0);
                            for (var G = 0; G < 4; G++) q ? d.set(W + G, k) : (P(h, M[G], N), d.set(W + G, N)), i && _.set(W + G, w[G]);
                            for (var G = 0; G < 6; G++) E[C++] = b[G] + W;
                            if (x && !q)
                                for (var G = 0; G < 2; G++) {
                                    for (var X = 3 * G, j = 0; j < 3; j++) {
                                        var Z = M[b[X] + j];
                                        P(h, Z, L[j])
                                    }
                                    o.sub(D, L[0], L[1]), o.sub(I, L[1], L[2]), o.cross(R, D, I);
                                    for (var j = 0; j < 3; j++) {
                                        var Y = 3 * M[b[X] + j];
                                        O[Y] = O[Y] + R[0], O[Y + 1] = O[Y + 1] + R[1], O[Y + 2] = O[Y + 2] + R[2]
                                    }
                                }
                        }
                    if (x)
                        for (var B = 0; B < O.length / 3; B++) P(O, B, R), o.normalize(R, R), O[3 * B] = R[0], O[3 * B + 1] = R[1], O[3 * B + 2] = R[2];
                    for (var U = [], K = [], B = 0; B < g - 1; B++)
                        for (var H = 0; H < v - 1; H++) {
                            var V = B * (v - 1) + H,
                                W = 4 * V;
                            S(B, H, M);
                            for (var G = 0; G < 4; G++) {
                                for (var j = 0; j < 4; j++) U[j] = F[4 * M[G] + j];
                                m.set(W + G, U), x && (P(O, M[G], R), f.set(W + G, R));
                                var Z = M[G];
                                K[0] = Z % v / (v - 1), K[1] = Math.floor(Z / v) / (g - 1), p.set(W + G, K)
                            }
                            V++
                        }
                } else {
                    for (var K = [], B = 0; B < u.count(); B++) {
                        K[0] = B % v / (v - 1), K[1] = Math.floor(B / v) / (g - 1);
                        var U = a.parseColor(u.getItemVisual(B, "color")),
                            z = u.getItemVisual(B, "opacity");
                        U[3] *= z, U[3] < .99 && (A = !0), m.set(B, U), p.set(B, K)
                    }
                    for (var M = [], Q = 0, B = 0; B < g - 1; B++)
                        for (var H = 0; H < v - 1; H++) { S(B, H, M); for (var G = 0; G < 6; G++) E[Q++] = M[b[G]] }
                    x ? s.generateVertexNormals() : f.value = null
                }
                e.material.get("normalMap") && s.generateTangents(), s.updateBoundingBox(), s.dirty(), e.material.transparent = A, e.material.depthMask = !A
            },
            _getDataShape: function(e, t) {
                for (var r = -1 / 0, n = 0, i = 0, a = t ? "u" : "x", o = 0; o < e.count(); o++) {
                    var s = e.get(a, o);
                    s < r && (i, i = 0, n++), r = s, i++
                }
                return { row: n + 1, column: i }
            },
            dispose: function() { this.groupGL.removeAll() },
            remove: function() { this.groupGL.removeAll() }
        })
    }, function(e, t, r) {
        r(0).registerLayout(function(e, t) {
            e.eachSeriesByType("surface", function(e) {
                var t = e.coordinateSystem;
                !t || t.type;
                var r = e.getData(),
                    n = new Float32Array(3 * r.count()),
                    i = [NaN, NaN, NaN];
                if (t && "cartesian3D" === t.type) {
                    var a = t.dimensions,
                        o = a.map(function(t) { return e.coordDimToDataDim(t)[0] });
                    r.each(o, function(e, a, o, s) {
                        var u;
                        u = r.hasValue(s) ? t.dataToPoint([e, a, o]) : i, n[3 * s] = u[0], n[3 * s + 1] = u[1], n[3 * s + 2] = u[2]
                    })
                }
                r.setLayout("points", n)
            })
        })
    }, function(e, t, r) {
        var n = r(0),
            i = r(38),
            a = r(32),
            o = r(31),
            s = r(33),
            u = r(64),
            h = n.extendComponentModel({
                type: "geo3D",
                layoutMode: "box",
                coordinateSystem: null,
                optionUpdated: function() {
                    var e = this.option;
                    e.regions = this.getFilledRegions(e.regions, e.map);
                    var t = n.helper.completeDimensions(["value"], e.data, { encodeDef: this.get("encode"), dimsDef: this.get("dimensions") }),
                        r = new n.List(t, this);
                    r.initData(e.regions);
                    var i = {};
                    r.each(function(e) {
                        var t = r.getName(e),
                            n = r.getItemModel(e);
                        i[t] = n
                    }), this._regionModelMap = i, this._data = r
                },
                getData: function() { return this._data },
                getRegionModel: function(e) { return this._regionModelMap[e] || new n.Model(null, this) },
                getFormattedLabel: function(e, t) {
                    var r = this._data.getName(e),
                        n = this.getRegionModel(r),
                        i = n.get("normal" === t ? ["label", "formatter"] : ["emphasis", "label", "formatter"]);
                    null == i && (i = n.get(["label", "formatter"]));
                    var a = { name: r };
                    if ("function" == typeof i) return a.status = t, i(a);
                    if ("string" == typeof i) { var o = a.seriesName; return i.replace("{a}", null != o ? o : "") }
                    return r
                },
                defaultOption: { regions: [] }
            });
        n.util.merge(h.prototype, u), n.util.merge(h.prototype, i), n.util.merge(h.prototype, a), n.util.merge(h.prototype, o), n.util.merge(h.prototype, s), e.exports = h
    }, function(e, t, r) {
        var n = r(61),
            i = r(0),
            a = r(2),
            o = r(40),
            s = r(30);
        e.exports = i.extendComponentView({
            type: "geo3D",
            __ecgl__: !0,
            init: function(e, t) { this._geo3DBuilder = new n(t), this.groupGL = new a.Node, this._lightRoot = new a.Node, this._sceneHelper = new s(this._lightRoot), this._sceneHelper.initLight(this._lightRoot), this._control = new o({ zr: t.getZr() }), this._control.init() },
            render: function(e, t, r) {
                this.groupGL.add(this._geo3DBuilder.rootNode);
                var n = e.coordinateSystem;
                if (n && n.viewGL) {
                    n.viewGL.add(this._lightRoot), e.get("show") ? n.viewGL.add(this.groupGL) : n.viewGL.remove(this.groupGL);
                    var i = this._control;
                    i.setViewGL(n.viewGL);
                    var a = e.getModel("viewControl");
                    i.setFromViewControlModel(a, 0), this._sceneHelper.setScene(n.viewGL.scene), this._sceneHelper.updateLight(e), n.viewGL.setPostEffect(e.getModel("postEffect"), r), n.viewGL.setTemporalSuperSampling(e.getModel("temporalSuperSampling")), this._geo3DBuilder.update(e, n, t, r);
                    var o = n.viewGL.isLinearSpace() ? "define" : "undefine";
                    this._geo3DBuilder.rootNode.traverse(function(e) { e.material && e.material.shader[o]("fragment", "SRGB_DECODE") }), i.off("update"), i.on("update", function() { r.dispatchAction({ type: "geo3DChangeCamera", alpha: i.getAlpha(), beta: i.getBeta(), distance: i.getDistance(), center: i.getCenter(), from: this.uid, geo3DId: e.id }) })
                }
            },
            afterRender: function(e, t, r, n) {
                var i = n.renderer;
                this._sceneHelper.updateAmbientCubemap(i, e, r), this._sceneHelper.updateSkybox(i, e, r)
            },
            dispose: function() { this._control.dispose() }
        })
    }, function(e, t, r) {
        function n(e, t) { e.id = e.id || e.name || t + "" }
        var i = r(0),
            a = r(38),
            o = r(32),
            s = r(31),
            u = r(33),
            h = i.extendComponentModel({
                type: "globe",
                layoutMode: "box",
                coordinateSystem: null,
                init: function() { h.superApply(this, "init", arguments), i.util.each(this.option.layers, function(e, t) { i.util.merge(e, this.defaultLayerOption), n(e, t) }, this) },
                mergeOption: function(e) {
                    function t(e) { return i.util.reduce(e, function(e, t, r) { return n(t, r), e[t.id] = t, e }, {}) }
                    var r = this.option.layers;
                    if (this.option.layers = null, h.superApply(this, "mergeOption", arguments), r && r.length) {
                        var a = t(e.layers),
                            o = t(r);
                        for (var s in a) o[s] ? i.util.merge(o[s], a[s], !0) : r.push(e.layers[s]);
                        this.option.layers = r
                    }
                    i.util.each(this.option.layers, function(e) { i.util.merge(e, this.defaultLayerOption) }, this)
                },
                optionUpdated: function() { this.updateDisplacementHash() },
                defaultLayerOption: { show: !0, type: "overlay" },
                defaultOption: { show: !0, zlevel: -10, left: 0, top: 0, width: "100%", height: "100%", environment: "auto", baseColor: "#fff", baseTexture: "", heightTexture: "", displacementTexture: "", displacementScale: 0, displacementQuality: "medium", globeRadius: 100, globeOuterRadius: 150, shading: "lambert", light: { main: { time: "" } }, viewControl: { autoRotate: !0, panSensitivity: 0, targetCoord: null }, layers: [] },
                getDisplacementTexture: function() { return this.get("displacementTexture") || this.get("heightTexture") },
                getDisplacemenScale: function() {
                    var e = this.getDisplacementTexture(),
                        t = this.get("displacementScale");
                    return e && "none" !== e || (t = 0), t
                },
                hasDisplacement: function() { return this.getDisplacemenScale() > 0 },
                _displacementChanged: !0,
                _displacementScale: 0,
                updateDisplacementHash: function() {
                    var e = this.getDisplacementTexture(),
                        t = this.getDisplacemenScale();
                    this._displacementChanged = this._displacementTexture !== e || this._displacementScale !== t, this._displacementTexture = e, this._displacementScale = t
                },
                isDisplacementChanged: function() { return this._displacementChanged }
            });
        i.util.merge(h.prototype, a), i.util.merge(h.prototype, o), i.util.merge(h.prototype, s), i.util.merge(h.prototype, u), e.exports = h
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(40),
            o = r(30),
            s = r(189),
            u = r(4);
        e.exports = n.extendComponentView({
            type: "globe",
            __ecgl__: !0,
            _displacementScale: 0,
            init: function(e, t) {
                this.groupGL = new i.Node;
                var r = {};
                i.COMMON_SHADERS.forEach(function(e) { r[e] = new i.Material({ shader: i.createShader("ecgl." + e) }) }), this._materials = r, this._sphereGeometry = new i.SphereGeometry({ widthSegments: 200, heightSegments: 100, dynamic: !0 }), this._overlayGeometry = new i.SphereGeometry({ widthSegments: 80, heightSegments: 40 }), this._planeGeometry = new i.PlaneGeometry, this._earthMesh = new i.Mesh({ renderNormal: !0 }), this._lightRoot = new i.Node, this._sceneHelper = new o, this._sceneHelper.initLight(this._lightRoot), this.groupGL.add(this._earthMesh), this._control = new a({ zr: t.getZr() }), this._control.init(), this._layerMeshes = {}
            },
            render: function(e, t, r) {
                var n = e.coordinateSystem,
                    a = e.get("shading");
                n.viewGL.add(this._lightRoot), e.get("show") ? n.viewGL.add(this.groupGL) : n.viewGL.remove(this.groupGL), this._sceneHelper.setScene(n.viewGL.scene), n.viewGL.setPostEffect(e.getModel("postEffect"), r), n.viewGL.setTemporalSuperSampling(e.getModel("temporalSuperSampling"));
                var o = this._earthMesh;
                o.geometry = this._sphereGeometry, this._materials[a] ? o.material = this._materials[a] : o.material = this._materials.lambert, i.setMaterialFromModel(a, o.material, e, r), ["roughnessMap", "metalnessMap", "detailMap", "normalMap"].forEach(function(e) {
                    var t = o.material.get(e);
                    t && (t.flipY = !1)
                }), o.material.set("color", i.parseColor(e.get("baseColor")));
                var s = .99 * n.radius;
                o.scale.set(s, s, s);
                var u = o.material.setTextureImage("diffuseMap", e.get("baseTexture"), r, { flipY: !1, anisotropic: 8 });
                u && u.surface && u.surface.attachToMesh(o);
                var h = o.material.setTextureImage("bumpMap", e.get("heightTexture"), r, { flipY: !1, anisotropic: 8 });
                h && h.surface && h.surface.attachToMesh(o), o.material.shader[e.get("postEffect.enable") ? "define" : "undefine"]("fragment", "SRGB_DECODE"), this._updateLight(e, r), this._displaceVertices(e, r), this._updateViewControl(e, r), this._updateLayers(e, r)
            },
            afterRender: function(e, t, r, n) {
                var i = n.renderer;
                this._sceneHelper.updateAmbientCubemap(i, e, r), this._sceneHelper.updateSkybox(i, e, r)
            },
            _updateLayers: function(e, t) {
                var r = e.coordinateSystem,
                    a = e.get("layers"),
                    o = r.radius,
                    s = [],
                    h = [],
                    l = [],
                    c = [];
                n.util.each(a, function(e) {
                    var a = new n.Model(e),
                        d = a.get("type"),
                        f = i.loadTexture(a.get("texture"), t, { flipY: !1, anisotropic: 8 });
                    if (f.surface && f.surface.attachToMesh(this._earthMesh), "blend" === d) {
                        var p = a.get("blendTo"),
                            _ = u.firstNotNull(a.get("intensity"), 1);
                        "emission" === p ? (l.push(f), c.push(_)) : (s.push(f), h.push(_))
                    } else {
                        var m = a.get("id"),
                            g = this._layerMeshes[m];
                        g || (g = this._layerMeshes[m] = new i.Mesh({ geometry: this._overlayGeometry, castShadow: !1, ignorePicking: !0 }));
                        "lambert" === a.get("shading") ? (g.material = g.__lambertMaterial || new i.Material({ shader: i.createShader("ecgl.lambert"), transparent: !0, depthMask: !1 }), g.__lambertMaterial = g.material) : (g.material = g.__colorMaterial || new i.Material({ shader: i.createShader("ecgl.color"), transparent: !0, depthMask: !1 }), g.__colorMaterial = g.material), g.material.shader.enableTexture("diffuseMap");
                        var v = a.get("distance"),
                            y = o + (null == v ? r.radius / 100 : v);
                        g.scale.set(y, y, y), o = y;
                        var x = this._blankTexture || (this._blankTexture = i.createBlankTexture("rgba(255, 255, 255, 0)"));
                        g.material.set("diffuseMap", x), i.loadTexture(a.get("texture"), t, { flipY: !1, anisotropic: 8 }, function(e) { e.surface && e.surface.attachToMesh(g), g.material.set("diffuseMap", e), t.getZr().refresh() }), a.get("show") ? this.groupGL.add(g) : this.groupGL.remove(g)
                    }
                }, this);
                var d = this._earthMesh.material;
                d.shader.define("fragment", "LAYER_DIFFUSEMAP_COUNT", s.length), d.shader.define("fragment", "LAYER_EMISSIVEMAP_COUNT", l.length), d.set("layerDiffuseMap", s), d.set("layerDiffuseIntensity", h), d.set("layerEmissiveMap", l), d.set("layerEmissionIntensity", c);
                var f = e.getModel("debug.wireframe");
                if (f.get("show")) {
                    d.shader.define("both", "WIREFRAME_TRIANGLE");
                    var p = i.parseColor(f.get("lineStyle.color") || "rgba(0,0,0,0.5)"),
                        _ = u.firstNotNull(f.get("lineStyle.width"), 1);
                    d.set("wireframeLineWidth", _), d.set("wireframeLineColor", p)
                } else d.shader.undefine("both", "WIREFRAME_TRIANGLE")
            },
            _updateViewControl: function(e, t) {
                function r() { return { type: "globeChangeCamera", alpha: a.getAlpha(), beta: a.getBeta(), distance: a.getDistance() - n.radius, center: a.getCenter(), from: this.uid, globeId: e.id } }
                var n = e.coordinateSystem,
                    i = e.getModel("viewControl"),
                    a = (n.viewGL.camera, this._control);
                a.setViewGL(n.viewGL);
                var o, s, u = i.get("targetCoord");
                null != u && (s = u[0] + 90, o = u[1]), a.setFromViewControlModel(i, { baseDistance: n.radius, alpha: o, beta: s }), a.off("update"), a.on("update", function() { t.dispatchAction(r()) })
            },
            _displaceVertices: function(e, t) {
                var r = e.get("displacementQuality"),
                    n = e.get("debug.wireframe.show"),
                    i = e.coordinateSystem;
                if (e.isDisplacementChanged() || r !== this._displacementQuality || n !== this._showDebugWireframe) {
                    this._displacementQuality = r, this._showDebugWireframe = n;
                    var a = this._sphereGeometry,
                        o = { low: 100, medium: 200, high: 400, ultra: 800 }[r] || 200,
                        s = o / 2;
                    (a.widthSegments !== o || n) && (a.widthSegments = o, a.heightSegments = s, a.build()), this._doDisplaceVertices(a, i), n && a.generateBarycentric()
                }
            },
            _doDisplaceVertices: function(e, t) {
                var r = e.attributes.position.value,
                    n = e.attributes.texcoord0.value,
                    i = e.__originalPosition;
                i && i.length === r.length || (i = new Float32Array(r.length), i.set(r), e.__originalPosition = i);
                for (var a = t.displacementWidth, o = t.displacementHeight, s = t.displacementData, u = 0; u < e.vertexCount; u++) {
                    var h = 3 * u,
                        l = 2 * u,
                        c = i[h + 1],
                        d = i[h + 2],
                        f = i[h + 3],
                        p = n[l++],
                        _ = n[l++],
                        m = Math.round(p * (a - 1)),
                        g = Math.round(_ * (o - 1)),
                        v = g * a + m,
                        y = s ? s[v] : 0;
                    r[h + 1] = c + c * y, r[h + 2] = d + d * y, r[h + 3] = f + f * y
                }
                e.generateVertexNormals(), e.dirty(), e.updateBoundingBox()
            },
            updateLayout: function(e, t, r) { this._displaceVertices(e, r) },
            _updateLight: function(e, t) {
                var r = this._earthMesh;
                this._sceneHelper.updateLight(e);
                var i = this._sceneHelper.mainLight,
                    a = e.get("light.main.time") || new Date,
                    o = s.getPosition(n.number.parseDate(a), 0, 0),
                    u = Math.cos(o.altitude);
                i.position.y = -u * Math.cos(o.azimuth), i.position.x = Math.sin(o.altitude), i.position.z = u * Math.sin(o.azimuth), i.lookAt(r.getWorldPosition())
            },
            dispose: function(e, t) { this.groupGL.removeAll(), this._control.dispose() }
        })
    }, function(e, t, r) {
        function n(e, t) { return t.type || (t.data ? "category" : "value") }
        var i = r(0),
            a = r(144),
            o = i.extendComponentModel({ type: "cartesian3DAxis", axis: null, getCoordSysModel: function() { return this.ecModel.queryComponents({ mainType: "grid3D", index: this.option.gridIndex, id: this.option.gridId })[0] } });
        i.helper.mixinAxisModelCommonMethods(o), a("x", o, n, { name: "X" }), a("y", o, n, { name: "Y" }), a("z", o, n, { name: "Z" })
    }, function(e, t, r) {
        function n(e, t) {
            var r = new a.Mesh({ geometry: new o({ useNativeLine: !1 }), material: t, castShadow: !1, ignorePicking: !0, renderOrder: 2 }),
                n = new u;
            n.material.depthMask = !1;
            var i = new a.Node;
            i.add(r), i.add(n), this.rootNode = i, this.dim = e, this.linesMesh = r, this.labelsMesh = n, this.axisLineCoords = null, this.labelElements = []
        }
        var i = r(0),
            a = r(2),
            o = r(22),
            s = r(4),
            u = r(51),
            h = s.firstNotNull,
            l = r(62),
            c = { x: 0, y: 2, z: 1 },
            d = { x: "y", y: "x", z: "y" };
        n.prototype.update = function(e, t, r, n) {
            var o = e.coordinateSystem,
                s = o.getAxis(this.dim),
                u = t[this.dim],
                f = this.linesMesh.geometry,
                p = this.labelsMesh.geometry;
            f.convertToDynamicArray(!0), p.convertToDynamicArray(!0);
            var _ = s.model,
                m = s.getExtent(),
                g = n.getDevicePixelRatio(),
                v = _.getModel("axisLine", e.getModel("axisLine")),
                y = _.getModel("axisTick", e.getModel("axisTick")),
                x = _.getModel("axisLabel", e.getModel("axisLabel")),
                T = v.get("lineStyle.color");
            if (v.get("show")) {
                var b = v.getModel("lineStyle"),
                    w = [0, 0, 0],
                    E = [0, 0, 0],
                    S = c[s.dim];
                w[S] = m[0], E[S] = m[1], this.axisLineCoords = [w, E];
                var A = a.parseColor(T),
                    M = h(b.get("width"), 1),
                    N = h(b.get("opacity"), 1);
                A[3] *= N, f.addLine(w, E, A, M * g)
            }
            if (y.get("show")) {
                var C = y.getModel("lineStyle"),
                    L = a.parseColor(h(C.get("color"), T)),
                    M = h(C.get("width"), 1);
                L[3] *= h(C.get("opacity"), 1);
                var D = s.getTicksCoords(),
                    I = y.get("interval");
                null != I && "auto" !== I || (I = u);
                for (var R = y.get("length"), P = 0; P < D.length; P++)
                    if (!l(s, P, I)) {
                        var O = D[P],
                            w = [0, 0, 0],
                            E = [0, 0, 0],
                            S = c[s.dim],
                            F = c[d[s.dim]];
                        w[S] = E[S] = O, E[F] = R, f.addLine(w, E, L, M * g)
                    }
            }
            this.labelElements = [];
            var g = n.getDevicePixelRatio();
            if (x.get("show"))
                for (var B = s.getLabelsCoords(), U = _.get("data"), I = u, z = x.get("margin"), G = _.getFormattedLabels(), k = s.scale.getTicks(), P = 0; P < B.length; P++)
                    if (!l(s, P, I)) {
                        var O = B[P],
                            H = [0, 0, 0],
                            S = c[s.dim],
                            F = c[d[s.dim]];
                        H[S] = H[S] = O, H[F] = z;
                        var V = x;
                        U && U[k[P]] && U[k[P]].textStyle && (V = new i.Model(U[k[P]].textStyle, x, _.ecModel));
                        var W = h(V.get("color"), T),
                            q = new i.graphic.Text;
                        i.graphic.setTextStyle(q.style, V, { text: G[P], textFill: "function" == typeof W ? W("category" === s.type ? G[P] : "value" === s.type ? k[P] + "" : k[P], P) : W, textVerticalAlign: "top", textAlign: "left" });
                        var X = r.add(q),
                            j = q.getBoundingRect();
                        p.addSprite(H, [j.width * g, j.height * g], X), this.labelElements.push(q)
                    }
            if (_.get("name")) {
                var Z = _.getModel("nameTextStyle"),
                    H = [0, 0, 0],
                    S = c[s.dim],
                    F = c[d[s.dim]],
                    Y = h(Z.get("color"), T),
                    K = Z.get("borderColor"),
                    M = Z.get("borderWidth");
                H[S] = H[S] = (m[0] + m[1]) / 2, H[F] = _.get("nameGap");
                var q = new i.graphic.Text;
                i.graphic.setTextStyle(q.style, Z, { text: _.get("name"), textFill: Y, textStroke: K, lineWidth: M });
                var X = r.add(q),
                    j = q.getBoundingRect();
                p.addSprite(H, [j.width * g, j.height * g], X), q.__idx = this.labelElements.length, this.nameLabelElement = q
            }
            this.labelsMesh.material.set("textureAtlas", r.getTexture()), this.labelsMesh.material.set("uvScale", r.getCoordsScale()), f.convertToTypedArray(), p.convertToTypedArray()
        }, n.prototype.setSpriteAlign = function(e, t, r) {
            for (var n = r.getDevicePixelRatio(), i = this.labelsMesh.geometry, a = 0; a < this.labelElements.length; a++) {
                var o = this.labelElements[a],
                    s = o.getBoundingRect();
                i.setSpriteAlign(a, [s.width * n, s.height * n], e, t)
            }
            var u = this.nameLabelElement;
            if (u) {
                var s = u.getBoundingRect();
                i.setSpriteAlign(u.__idx, [s.width * n, s.height * n], e, t), i.dirty()
            }
            this.textAlign = e, this.textVerticalAlign = t
        }, e.exports = n
    }, function(e, t, r) {
        function n(e, t, r, n) {
            var i = [0, 0, 0],
                a = n < 0 ? r.getExtentMin() : r.getExtentMax();
            i[d[r.dim]] = a, e.position.setArray(i), e.rotation.identity(), t.distance = -Math.abs(a), t.normal.set(0, 0, 0), "x" === r.dim ? (e.rotation.rotateY(n * Math.PI / 2), t.normal.x = -n) : "z" === r.dim ? (e.rotation.rotateX(-n * Math.PI / 2), t.normal.y = -n) : (n > 0 && e.rotation.rotateY(Math.PI), t.normal.z = -n)
        }

        function i(e, t, r) {
            this.rootNode = new o.Node;
            var n = new o.Mesh({ geometry: new u({ useNativeLine: !1 }), material: t, castShadow: !1, ignorePicking: !0, renderOrder: 1 }),
                i = new o.Mesh({ geometry: new h, material: r, castShadow: !1, culling: !1, ignorePicking: !0, renderOrder: 0 });
            this.rootNode.add(i), this.rootNode.add(n), this.faceInfo = e, this.plane = new o.Plane, this.linesMesh = n, this.quadsMesh = i
        }
        var a = r(0),
            o = r(2),
            s = r(4),
            u = r(22),
            h = r(175),
            l = s.firstNotNull,
            c = r(62),
            d = { x: 0, y: 2, z: 1 };
        i.prototype.update = function(e, t, r, i) {
            var a = t.coordinateSystem,
                o = [a.getAxis(this.faceInfo[0]), a.getAxis(this.faceInfo[1])],
                s = this.linesMesh.geometry,
                u = this.quadsMesh.geometry;
            s.convertToDynamicArray(!0), u.convertToDynamicArray(!0), this._updateSplitLines(s, o, t, e, i), this._udpateSplitAreas(u, o, t, e, i), s.convertToTypedArray(), u.convertToTypedArray();
            var h = a.getAxis(this.faceInfo[2]);
            n(this.rootNode, this.plane, h, this.faceInfo[3])
        }, i.prototype._updateSplitLines = function(e, t, r, n, i) {
            var s = i.getDevicePixelRatio();
            t.forEach(function(i, u) {
                var h = i.model,
                    d = t[1 - u].getExtent();
                if (!i.scale.isBlank()) {
                    var f = h.getModel("splitLine", r.getModel("splitLine"));
                    if (f.get("show")) {
                        var p = f.getModel("lineStyle"),
                            _ = p.get("color"),
                            m = l(p.get("opacity"), 1),
                            g = l(p.get("width"), 1),
                            v = f.get("interval");
                        null != v && "auto" !== v || (v = n[i.dim]), _ = a.util.isArray(_) ? _ : [_];
                        for (var y = i.getTicksCoords(), x = 0, T = 0; T < y.length; T++)
                            if (!c(i, T, v)) {
                                var b = y[T],
                                    w = o.parseColor(_[x % _.length]);
                                w[3] *= m;
                                var E = [0, 0, 0],
                                    S = [0, 0, 0];
                                E[u] = S[u] = b, E[1 - u] = d[0], S[1 - u] = d[1], e.addLine(E, S, w, g * s), x++
                            }
                    }
                }
            })
        }, i.prototype._udpateSplitAreas = function(e, t, r, n, i) {
            t.forEach(function(i, s) {
                var u = i.model,
                    h = t[1 - s].getExtent();
                if (!i.scale.isBlank()) {
                    var d = u.getModel("splitArea", r.getModel("splitArea"));
                    if (d.get("show")) {
                        var f = d.getModel("areaStyle"),
                            p = f.get("color"),
                            _ = l(f.get("opacity"), 1),
                            m = d.get("interval");
                        null != m && "auto" !== m || (m = n[i.dim]), p = a.util.isArray(p) ? p : [p];
                        for (var g = i.getTicksCoords(), v = 0, y = [0, 0, 0], x = [0, 0, 0], T = 0; T < g.length; T++) {
                            var b = g[T],
                                w = [0, 0, 0],
                                E = [0, 0, 0];
                            if (w[s] = E[s] = b, w[1 - s] = h[0], E[1 - s] = h[1], 0 !== T) {
                                if (!c(i, T, m)) {
                                    var S = o.parseColor(p[v % p.length]);
                                    S[3] *= _, e.addQuad([y, w, E, x], S), y = w, x = E, v++
                                }
                            } else y = w, x = E
                        }
                    }
                }
            })
        }, e.exports = i
    }, function(e, t, r) {
        var n = r(0),
            i = r(38),
            a = r(32),
            o = r(31),
            s = n.extendComponentModel({ type: "grid3D", dependencies: ["xAxis3D", "yAxis3D", "zAxis3D"], defaultOption: { show: !0, zlevel: -10, left: 0, top: 0, width: "100%", height: "100%", environment: "auto", boxWidth: 100, boxHeight: 100, boxDepth: 100, axisPointer: { show: !0, lineStyle: { color: "rgba(0, 0, 0, 0.8)", width: 1 }, label: { show: !0, formatter: null, margin: 8, textStyle: { fontSize: 14, color: "#fff", backgroundColor: "rgba(0,0,0,0.5)", padding: 3, borderRadius: 3 } } }, axisLine: { show: !0, lineStyle: { color: "#333", width: 2, type: "solid" } }, axisTick: { show: !0, inside: !1, length: 3, lineStyle: { width: 1 } }, axisLabel: { show: !0, inside: !1, rotate: 0, margin: 8, textStyle: { fontSize: 12 } }, splitLine: { show: !0, lineStyle: { color: ["#ccc"], width: 1, type: "solid" } }, splitArea: { show: !1, areaStyle: { color: ["rgba(250,250,250,0.3)", "rgba(200,200,200,0.3)"] } }, light: { main: { alpha: 30, beta: 40 }, ambient: { intensity: .4 } }, viewControl: { alpha: 20, beta: 40, autoRotate: !1, distance: 200, minDistance: 40, maxDistance: 400 } } });
        n.util.merge(s.prototype, i), n.util.merge(s.prototype, a), n.util.merge(s.prototype, o), e.exports = s
    }, function(e, t, r) {
        var n = r(0),
            i = r(2),
            a = r(40),
            o = r(22),
            s = r(4),
            u = s.firstNotNull,
            h = r(67),
            l = r(30),
            c = r(140),
            d = r(139),
            f = r(51);
        i.Shader.import(r(41)), ["x", "y", "z"].forEach(function(e) { n.extendComponentView({ type: e + "Axis3D" }) });
        var p = { x: 0, y: 2, z: 1 };
        e.exports = n.extendComponentView({
            type: "grid3D",
            __ecgl__: !0,
            init: function(e, t) {
                var r = [
                        ["y", "z", "x", -1, "left"],
                        ["y", "z", "x", 1, "right"],
                        ["x", "y", "z", -1, "bottom"],
                        ["x", "y", "z", 1, "top"],
                        ["x", "z", "y", -1, "far"],
                        ["x", "z", "y", 1, "near"]
                    ],
                    n = ["x", "y", "z"],
                    s = new i.Material({ shader: i.createShader("ecgl.color"), depthMask: !1, transparent: !0 }),
                    u = new i.Material({ shader: i.createShader("ecgl.meshLines3D"), depthMask: !1, transparent: !0 });
                s.shader.define("fragment", "DOUBLE_SIDED"), s.shader.define("both", "VERTEX_COLOR"), this.groupGL = new i.Node, this._control = new a({ zr: t.getZr() }), this._control.init(), this._faces = r.map(function(e) { var t = new c(e, u, s); return this.groupGL.add(t.rootNode), t }, this), this._axes = n.map(function(e) { var t = new d(e, u); return this.groupGL.add(t.rootNode), t }, this);
                var p = t.getDevicePixelRatio();
                this._axisLabelSurface = new h({ width: 256, height: 256, devicePixelRatio: p }), this._axisLabelSurface.onupdate = function() { t.getZr().refresh() }, this._axisPointerLineMesh = new i.Mesh({ geometry: new o({ useNativeLine: !1 }), material: u, castShadow: !1, ignorePicking: !0, renderOrder: 3 }), this.groupGL.add(this._axisPointerLineMesh), this._axisPointerLabelsSurface = new h({ width: 128, height: 128, devicePixelRatio: p }), this._axisPointerLabelsMesh = new f({ ignorePicking: !0, renderOrder: 4, castShadow: !1 }), this._axisPointerLabelsMesh.material.set("textureAtlas", this._axisPointerLabelsSurface.getTexture()), this.groupGL.add(this._axisPointerLabelsMesh), this._lightRoot = new i.Node, this._sceneHelper = new l, this._sceneHelper.initLight(this._lightRoot)
            },
            render: function(e, t, r) {
                this._model = e, this._api = r;
                var n = e.coordinateSystem;
                n.viewGL.add(this._lightRoot), e.get("show") ? n.viewGL.add(this.groupGL) : n.viewGL.remove(this.groupGL);
                var i = this._control;
                i.setViewGL(n.viewGL);
                var a = e.getModel("viewControl");
                i.setFromViewControlModel(a, 0), this._axisLabelSurface.clear();
                var o = ["x", "y", "z"].reduce(function(t, r) {
                    var i = n.getAxis(r),
                        a = i.model;
                    return t[r] = u(a.get("axisLabel.interval"), e.get("axisLabel.interval")), "ordinal" === i.scale.type && (null != t[r] && "auto" != t[r] || (t[r] = Math.floor(i.scale.getTicks().length / 8))), t
                }, {});
                i.off("update"), e.get("show") && (this._faces.forEach(function(n) { n.update(o, e, t, r) }, this), this._axes.forEach(function(t) { t.update(e, o, this._axisLabelSurface, r) }, this)), i.on("update", this._onCameraChange.bind(this, e, r), this), this._sceneHelper.setScene(n.viewGL.scene), this._sceneHelper.updateLight(e), n.viewGL.setPostEffect(e.getModel("postEffect"), r), n.viewGL.setTemporalSuperSampling(e.getModel("temporalSuperSampling")), this._initMouseHandler(e)
            },
            afterRender: function(e, t, r, n) {
                var i = n.renderer;
                this._sceneHelper.updateAmbientCubemap(i, e, r), this._sceneHelper.updateSkybox(i, e, r)
            },
            showAxisPointer: function(e, t, r, n) { this._doShowAxisPointer(), this._updateAxisPointer(n.value) },
            hideAxisPointer: function(e, t, r, n) { this._doHideAxisPointer() },
            _initMouseHandler: function(e) {
                var t = e.coordinateSystem,
                    r = t.viewGL;
                e.get("show") && e.get("axisPointer.show") ? r.on("mousemove", this._updateAxisPointerOnMousePosition, this) : r.off("mousemove", this._updateAxisPointerOnMousePosition)
            },
            _updateAxisPointerOnMousePosition: function(e) {
                if (!e.target) {
                    for (var t, r = this._model, n = r.coordinateSystem, a = n.viewGL, o = a.castRay(e.offsetX, e.offsetY, new i.Ray), s = 0; s < this._faces.length; s++) {
                        var u = this._faces[s];
                        if (!u.rootNode.invisible) {
                            u.plane.normal.dot(a.camera.worldTransform.z) < 0 && u.plane.normal.negate();
                            var h = o.intersectPlane(u.plane);
                            if (h) {
                                var l = n.getAxis(u.faceInfo[0]),
                                    c = n.getAxis(u.faceInfo[1]),
                                    d = p[u.faceInfo[0]],
                                    f = p[u.faceInfo[1]];
                                l.contain(h._array[d]) && c.contain(h._array[f]) && (t = h)
                            }
                        }
                    }
                    if (t) {
                        var _ = n.pointToData(t._array, [], !0);
                        this._updateAxisPointer(_), this._doShowAxisPointer()
                    } else this._doHideAxisPointer()
                }
            },
            _onCameraChange: function(e, t) {
                e.get("show") && (this._updateFaceVisibility(), this._updateAxisLinePosition());
                var r = this._control;
                t.dispatchAction({ type: "grid3DChangeCamera", alpha: r.getAlpha(), beta: r.getBeta(), distance: r.getDistance(), center: r.getCenter(), from: this.uid, grid3DId: e.id })
            },
            _updateFaceVisibility: function() {
                var e = this._control.getCamera(),
                    t = new i.Vector3;
                e.update();
                for (var r = 0; r < this._faces.length / 2; r++) {
                    for (var n = [], a = 0; a < 2; a++) { this._faces[2 * r + a].rootNode.getWorldPosition(t), t.transformMat4(e.viewMatrix), n[a] = t.z }
                    var o = n[0] > n[1] ? 0 : 1,
                        s = this._faces[2 * r + o],
                        u = this._faces[2 * r + 1 - o];
                    s.rootNode.invisible = !0, u.rootNode.invisible = !1
                }
            },
            _updateAxisLinePosition: function() {
                var e = this._model.coordinateSystem,
                    t = e.getAxis("x"),
                    r = e.getAxis("y"),
                    n = e.getAxis("z"),
                    i = n.getExtentMax(),
                    a = n.getExtentMin(),
                    o = t.getExtentMin(),
                    s = t.getExtentMax(),
                    u = r.getExtentMax(),
                    h = r.getExtentMin(),
                    l = this._axes[0].rootNode,
                    c = this._axes[1].rootNode,
                    d = this._axes[2].rootNode,
                    f = this._faces,
                    p = f[4].rootNode.invisible ? h : u,
                    _ = f[2].rootNode.invisible ? i : a,
                    m = f[0].rootNode.invisible ? o : s,
                    g = f[2].rootNode.invisible ? i : a,
                    v = f[0].rootNode.invisible ? s : o,
                    y = f[4].rootNode.invisible ? h : u;
                l.rotation.identity(), c.rotation.identity(), d.rotation.identity(), f[4].rootNode.invisible && (this._axes[0].flipped = !0, l.rotation.rotateX(Math.PI)), f[0].rootNode.invisible && (this._axes[1].flipped = !0, c.rotation.rotateZ(Math.PI)), f[4].rootNode.invisible && (this._axes[2].flipped = !0, d.rotation.rotateY(Math.PI)), l.position.set(0, _, p), c.position.set(m, g, 0), d.position.set(v, 0, y), l.update(), c.update(), d.update(), this._updateAxisLabelAlign()
            },
            _updateAxisLabelAlign: function() {
                var e = this._control.getCamera(),
                    t = [new i.Vector4, new i.Vector4],
                    r = new i.Vector4;
                this.groupGL.getWorldPosition(r), r.w = 1, r.transformMat4(e.viewMatrix).transformMat4(e.projectionMatrix), r.x /= r.w, r.y /= r.w, this._axes.forEach(function(n) {
                    for (var i = n.axisLineCoords, a = (n.labelsMesh.geometry, 0); a < t.length; a++) t[a].setArray(i[a]), t[a].w = 1, t[a].transformMat4(n.rootNode.worldTransform).transformMat4(e.viewMatrix).transformMat4(e.projectionMatrix), t[a].x /= t[a].w, t[a].y /= t[a].w;
                    var o, s, u = t[1].x - t[0].x,
                        h = t[1].y - t[0].y,
                        l = (t[1].x + t[0].x) / 2,
                        c = (t[1].y + t[0].y) / 2;
                    Math.abs(h / u) < .5 ? (o = "center", s = c > r.y ? "bottom" : "top") : (s = "middle", o = l > r.x ? "left" : "right"), n.setSpriteAlign(o, s, this._api)
                }, this)
            },
            _doShowAxisPointer: function() { this._axisPointerLineMesh.invisible && (this._axisPointerLineMesh.invisible = !1, this._axisPointerLabelsMesh.invisible = !1, this._api.getZr().refresh()) },
            _doHideAxisPointer: function() { this._axisPointerLineMesh.invisible || (this._axisPointerLineMesh.invisible = !0, this._axisPointerLabelsMesh.invisible = !0, this._api.getZr().refresh()) },
            _updateAxisPointer: function(e) {
                function t(e) { return s.firstNotNull(e.model.get("axisPointer.show"), l.get("show")) }

                function r(e) {
                    var t = e.model.getModel("axisPointer", l),
                        r = t.getModel("lineStyle"),
                        n = i.parseColor(r.get("color")),
                        a = u(r.get("width"), 1),
                        o = u(r.get("opacity"), 1);
                    return n[3] *= o, { color: n, lineWidth: a }
                }
                var n = this._model.coordinateSystem,
                    a = n.dataToPoint(e),
                    o = this._axisPointerLineMesh,
                    h = o.geometry,
                    l = this._model.getModel("axisPointer"),
                    c = this._api.getDevicePixelRatio();
                h.convertToDynamicArray(!0);
                for (var d = 0; d < this._faces.length; d++) {
                    var f = this._faces[d];
                    if (!f.rootNode.invisible) {
                        for (var _ = f.faceInfo, m = _[3] < 0 ? n.getAxis(_[2]).getExtentMin() : n.getAxis(_[2]).getExtentMax(), g = p[_[2]], v = 0; v < 2; v++) {
                            var y = _[v],
                                x = _[1 - v],
                                T = n.getAxis(y),
                                b = n.getAxis(x);
                            if (t(T)) {
                                var w = [0, 0, 0],
                                    E = [0, 0, 0],
                                    S = p[y],
                                    A = p[x];
                                w[S] = E[S] = a[S], w[g] = E[g] = m, w[A] = b.getExtentMin(), E[A] = b.getExtentMax();
                                var M = r(T);
                                h.addLine(w, E, M.color, M.lineWidth * c)
                            }
                        }
                        if (t(n.getAxis(_[2]))) {
                            var w = a.slice(),
                                E = a.slice();
                            E[g] = m;
                            var M = r(n.getAxis(_[2]));
                            h.addLine(w, E, M.color, M.lineWidth * c)
                        }
                    }
                }
                h.convertToTypedArray(), this._updateAxisPointerLabelsMesh(e), this._api.getZr().refresh()
            },
            _updateAxisPointerLabelsMesh: function(e) {
                var t = this._model,
                    r = this._axisPointerLabelsMesh,
                    i = this._axisPointerLabelsSurface,
                    a = t.coordinateSystem,
                    o = t.getModel("axisPointer");
                r.geometry.convertToDynamicArray(!0), i.clear();
                var s = { x: "y", y: "x", z: "y" };
                this._axes.forEach(function(t, u) {
                    var h = a.getAxis(t.dim),
                        l = h.model,
                        c = l.getModel("axisPointer", o),
                        d = c.getModel("label"),
                        f = c.get("lineStyle.color");
                    if (d.get("show") && c.get("show")) {
                        var _ = e[u],
                            m = d.get("formatter"),
                            g = h.scale.getLabel(_);
                        if (null != m) g = m(g, e);
                        else if ("interval" === h.scale.type || "log" === h.scale.type) {
                            var v = n.number.getPrecisionSafe(h.scale.getTicks()[0]);
                            g = _.toFixed(v + 2)
                        }
                        var y = d.getModel("textStyle"),
                            x = y.get("color"),
                            T = new n.graphic.Text;
                        n.graphic.setTextStyle(T.style, y, { text: g, textFill: x || f, textAlign: "left", textVerticalAlign: "top" });
                        var b = i.add(T),
                            w = T.getBoundingRect(),
                            E = this._api.getDevicePixelRatio(),
                            S = t.rootNode.position.toArray();
                        S[p[s[t.dim]]] += (t.flipped ? -1 : 1) * d.get("margin"), S[p[t.dim]] = h.dataToCoord(e[u]), r.geometry.addSprite(S, [w.width * E, w.height * E], b, t.textAlign, t.textVerticalAlign)
                    }
                }, this), i.getZr().refreshImmediately(), r.material.set("uvScale", i.getCoordsScale()), r.geometry.convertToTypedArray()
            },
            dispose: function() { this.groupGL.removeAll(), this._control.dispose() }
        })
    }, function(e, t, r) {
        var n = r(0),
            i = { show: !0, grid3DIndex: 0, inverse: !1, name: "", nameLocation: "middle", nameTextStyle: { fontSize: 16 }, nameGap: 20, axisPointer: {}, axisLine: {}, axisTick: {}, axisLabel: {}, splitArea: {} },
            a = n.util.merge({ boundaryGap: !0, axisTick: { alignWithLabel: !1, interval: "auto" }, axisLabel: { interval: "auto" }, axisPointer: { label: { show: !1 } } }, i),
            o = n.util.merge({ boundaryGap: [0, 0], splitNumber: 5, axisPointer: { label: {} } }, i),
            s = n.util.defaults({ scale: !0, min: "dataMin", max: "dataMax" }, o),
            u = n.util.defaults({ logBase: 10 }, o);
        u.scale = !0, e.exports = { categoryAxis: a, valueAxis: o, timeAxis: s, logAxis: u }
    }, function(e, t, r) {
        var n = r(0),
            i = r(143),
            a = ["value", "category", "time", "log"];
        e.exports = function(e, t, r, o) {
            n.util.each(a, function(a) {
                t.extend({
                    type: e + "Axis3D." + a,
                    mergeDefaultAndTheme: function(t, i) {
                        var o = i.getTheme();
                        n.util.merge(t, o.get(a + "Axis")), n.util.merge(t, this.getDefaultOption()), t.type = r(e, t)
                    },
                    defaultOption: n.util.merge(n.util.clone(i[a + "Axis"]), o || {}, !0)
                })
            }), t.superClass.registerSubTypeDefaulter(e + "Axis3D", n.util.curry(r, e))
        }
    }, function(e, t, r) {
        function n(e, t) {
            if (this.id = e, this.zr = t, this.dom = document.createElement("div"), this.dom.style.cssText = "position:absolute;left:0;right:0;top:0;bottom:0;", !mapboxgl) throw new Error("Mapbox GL library must be included. See https://www.mapbox.com/mapbox-gl-js/api/");
            this._mapbox = new mapboxgl.Map({ container: this.dom }), this._initEvents()
        }
        r(0);
        n.prototype.resize = function() { this._mapbox.resize() }, n.prototype.getMapbox = function() { return this._mapbox }, n.prototype.clear = function() {}, n.prototype.refresh = function() { this._mapbox.resize() };
        var i = ["mousedown", "mouseup", "click", "dblclick", "mousemove", "mousewheel", "wheel", "touchstart", "touchend", "touchmove", "touchcancel"];
        n.prototype._initEvents = function() {
            var e = this._mapbox.getCanvasContainer();
            this._handlers = this._handlers || { contextmenu: function(e) { return e.preventDefault(), !1 } }, i.forEach(function(t) {
                this._handlers[t] = function(t) {
                    var r = {};
                    for (var n in t) r[n] = t[n];
                    r.bubbles = !1;
                    var i = new t.constructor(t.type, r);
                    e.dispatchEvent(i)
                }, this.zr.dom.addEventListener(t, this._handlers[t])
            }, this), this.zr.dom.addEventListener("contextmenu", this._handlers.contextmenu)
        }, n.prototype.dispose = function() { i.forEach(function(e) { this.zr.dom.removeEventListener(e, this._handlers[e]) }, this) }, e.exports = n
    }, function(e, t, r) {
        var n = r(0),
            i = r(32),
            a = r(31),
            o = ["zoom", "center", "pitch", "bearing"],
            s = n.extendComponentModel({ type: "mapbox", layoutMode: "box", coordinateSystem: null, defaultOption: { zlevel: -10, style: "mapbox://styles/mapbox/light-v9", center: [0, 0], zoom: 0, pitch: 0, bearing: 0, light: { main: { alpha: 20, beta: 30 } }, altitudeScale: 1, boxHeight: "auto" }, getMapboxCameraOption: function() { var e = this; return o.reduce(function(t, r) { return t[r] = e.get(r), t }, {}) }, setMapboxCameraOption: function(e) { null != e && o.forEach(function(t) { null != e[t] && (this.option[t] = e[t]) }, this) }, getMapbox: function() { return this._mapbox }, setMapbox: function(e) { this._mapbox = e } });
        n.util.merge(s.prototype, i), n.util.merge(s.prototype, a), e.exports = s
    }, function(e, t, r) {
        var n = r(0),
            i = r(145),
            a = r(30),
            o = r(2);
        o.Shader.import(r(180));
        e.exports = n.extendComponentView({
            type: "mapbox",
            __ecgl__: !0,
            init: function(e, t) {
                var r = t.getZr();
                this._zrLayer = new i("mapbox", r), r.painter.insertLayer(-1e3, this._zrLayer), this._lightRoot = new o.Node, this._sceneHelper = new a(this._lightRoot), this._sceneHelper.initLight(this._lightRoot);
                var n = this._zrLayer.getMapbox(),
                    s = this._dispatchInteractAction.bind(this, t, n);
                ["zoom", "rotate", "drag", "pitch", "rotate", "move"].forEach(function(e) { n.on(e, s) }), this._groundMesh = new o.Mesh({ geometry: new o.PlaneGeometry, material: new o.Material({ shader: new o.Shader({ vertex: o.Shader.source("ecgl.displayShadow.vertex"), fragment: o.Shader.source("ecgl.displayShadow.fragment") }), depthMask: !1 }), renderOrder: -100, culling: !1, castShadow: !1, $ignorePicking: !0, renderNormal: !0 })
            },
            render: function(e, t, r) {
                var n = this._zrLayer.getMapbox(),
                    i = e.get("style"),
                    a = JSON.stringify(i);
                a !== this._oldStyleStr && i && n.setStyle(i), this._oldStyleStr = a, n.setCenter(e.get("center")), n.setZoom(e.get("zoom")), n.setPitch(e.get("pitch")), n.setBearing(e.get("bearing")), e.setMapbox(n);
                var o = e.coordinateSystem;
                o.viewGL.scene.add(this._lightRoot), o.viewGL.add(this._groundMesh), this._updateGroundMesh(), this._sceneHelper.setScene(o.viewGL.scene), this._sceneHelper.updateLight(e), o.viewGL.setPostEffect(e.getModel("postEffect"), r), o.viewGL.setTemporalSuperSampling(e.getModel("temporalSuperSampling")), this._mapboxModel = e
            },
            afterRender: function(e, t, r, n) {
                var i = n.renderer;
                this._sceneHelper.updateAmbientCubemap(i, e, r), this._sceneHelper.updateSkybox(i, e, r), e.coordinateSystem.viewGL.scene.traverse(function(e) { e.material && (e.material.shader.define("fragment", "NORMAL_UP_AXIS", 2), e.material.shader.define("fragment", "NORMAL_FRONT_AXIS", 1)) })
            },
            updateCamera: function(e, t, r, n) { e.coordinateSystem.setCameraOption(n), this._updateGroundMesh(), r.getZr().refresh() },
            _dispatchInteractAction: function(e, t, r) { e.dispatchAction({ type: "mapboxChangeCamera", pitch: t.getPitch(), zoom: t.getZoom(), center: t.getCenter().toArray(), bearing: t.getBearing(), mapboxId: this._mapboxModel && this._mapboxModel.id }) },
            _updateGroundMesh: function() {
                if (this._mapboxModel) {
                    var e = this._mapboxModel.coordinateSystem,
                        t = e.dataToPoint(e.center);
                    this._groundMesh.position.set(t[0], t[1], -.001);
                    var r = new o.Plane(new o.Vector3(0, 0, 1), 0),
                        n = e.viewGL.camera.castRay(new o.Vector2(-1, -1)),
                        i = e.viewGL.camera.castRay(new o.Vector2(1, 1)),
                        a = n.intersectPlane(r),
                        s = i.intersectPlane(r),
                        u = a.dist(s) / e.viewGL.rootNode.scale.x;
                    this._groundMesh.scale.set(u, u, 1)
                }
            },
            dispose: function(e, t) { t.getZr().delLayer(-1e3) }
        })
    }, function(e, t, r) {
        function n(e) { this.radius = e, this.viewGL = null, this.altitudeAxis, this.displacementData = null, this.displacementWidth, this.displacementHeight }
        var i = r(1),
            a = i.vec3;
        n.prototype = {
            constructor: n,
            dimensions: ["lng", "lat", "alt"],
            type: "globe",
            containPoint: function() {},
            setDisplacementData: function(e, t, r) { this.displacementData = e, this.displacementWidth = t, this.displacementHeight = r },
            _getDisplacementScale: function(e, t) {
                var r = (e + 180) / 360 * (this.displacementWidth - 1),
                    n = (90 - t) / 180 * (this.displacementHeight - 1),
                    i = Math.round(r) + Math.round(n) * this.displacementWidth;
                return this.displacementData[i]
            },
            dataToPoint: function(e, t) {
                var r = e[0],
                    n = e[1],
                    i = e[2] || 0,
                    a = this.radius;
                this.displacementData && (a *= 1 + this._getDisplacementScale(r, n)), this.altitudeAxis && (a += this.altitudeAxis.dataToCoord(i)), r = r * Math.PI / 180, n = n * Math.PI / 180;
                var o = Math.cos(n) * a;
                return t = t || [], t[0] = -o * Math.cos(r + Math.PI), t[1] = Math.sin(n) * a, t[2] = o * Math.sin(r + Math.PI), t
            },
            pointToData: function(e, t) {
                var r = e[0],
                    n = e[1],
                    i = e[2],
                    o = a.len(e);
                r /= o, n /= o, i /= o;
                var s = Math.asin(n),
                    u = Math.atan2(i, -r);
                u < 0 && (u = 2 * Math.PI + u);
                var h = 180 * s / Math.PI,
                    l = 180 * u / Math.PI - 180;
                return t = t || [], t[0] = l, t[1] = h, t[2] = o - this.radius, this.altitudeAxis && (t[2] = this.altitudeAxis.coordToData(t[2])), t
            }
        }, e.exports = n
    }, function(e, t, r) {
        function n(e, t) {
            var r = document.createElement("canvas"),
                n = r.getContext("2d"),
                i = e.width,
                a = e.height;
            r.width = i, r.height = a, n.drawImage(e, 0, 0, i, a);
            for (var o = n.getImageData(0, 0, i, a).data, s = new Float32Array(o.length / 4), u = 0; u < o.length / 4; u++) {
                var h = o[4 * u];
                s[u] = h / 255 * t
            }
            return { data: s, width: i, height: a }
        }

        function i(e, t) {
            var r = e.getBoxLayoutParams(),
                n = u.getLayoutRect(r, { width: t.getWidth(), height: t.getHeight() });
            n.y = t.getHeight() - n.y - n.height, this.viewGL.setViewport(n.x, n.y, n.width, n.height, t.getDevicePixelRatio()), this.radius = e.get("globeRadius");
            var i = e.get("globeOuterRadius");
            this.altitudeAxis && this.altitudeAxis.setExtent(0, i - this.radius)
        }

        function a(e, t) {
            var r = [1 / 0, -1 / 0];
            if (e.eachSeries(function(e) {
                    if (e.coordinateSystem === this) {
                        var t = e.getData(),
                            n = e.coordDimToDataDim("alt")[0];
                        if (n) {
                            var i = t.getDataExtent(n, !0);
                            r[0] = Math.min(r[0], i[0]), r[1] = Math.max(r[1], i[1])
                        }
                    }
                }, this), r && isFinite(r[1] - r[0])) {
                var n = s.helper.createScale(r, { type: "value", min: "dataMin", max: "dataMax" });
                this.altitudeAxis = new s.Axis("altitude", n), this.resize(this.model, t)
            }
        }
        var o = r(148),
            s = r(0),
            u = r(42),
            h = r(21),
            l = r(4),
            c = r(2),
            d = {
                dimensions: o.prototype.dimensions,
                create: function(e, t) {
                    var r = [];
                    return e.eachComponent("globe", function(e) {
                        e.__viewGL = e.__viewGL || new h;
                        var n = new o;
                        n.viewGL = e.__viewGL, e.coordinateSystem = n, n.model = e, r.push(n), n.resize = i, n.resize(e, t), n.update = a
                    }), e.eachSeries(function(t) {
                        if ("globe" === t.get("coordinateSystem")) {
                            var r = t.getReferringComponents("globe")[0];
                            if (r || (r = e.getComponent("globe")), !r) throw new Error('globe "' + l.firstNotNull(t.get("globe3DIndex"), t.get("globe3DId"), 0) + '" not found');
                            var n = r.coordinateSystem;
                            t.coordinateSystem = n
                        }
                    }), e.eachComponent("globe", function(e, r) {
                        var i = e.coordinateSystem,
                            a = e.getDisplacementTexture(),
                            o = e.getDisplacemenScale();
                        if (e.isDisplacementChanged())
                            if (e.hasDisplacement()) {
                                var s = !0;
                                c.loadTexture(a, t, function(e) {
                                    var r = e.image,
                                        a = n(r, o);
                                    i.setDisplacementData(a.data, a.width, a.height), s || t.dispatchAction({ type: "globeUpdateDisplacment" })
                                }), s = !1
                            } else i.setDisplacementData(null, 0, 0)
                    }), r
                }
            };
        s.registerCoordinateSystem("globe", d), e.exports = d
    }, function(e, t, r) {
        function n(e, t, r) { i.Axis.call(this, e, t, r) }
        var i = r(0);
        n.prototype = { constructor: n, getExtentMin: function() { var e = this._extent; return Math.min(e[0], e[1]) }, getExtentMax: function() { var e = this._extent; return Math.max(e[0], e[1]) } }, i.util.inherits(n, i.Axis), e.exports = n
    }, function(e, t, r) {
        function n(e) { a.call(this, e), this.size = [0, 0, 0] }
        var i = r(0),
            a = r(190);
        n.prototype = { constructor: n, type: "cartesian3D", dimensions: ["x", "y", "z"], model: null, containPoint: function(e) { return this.getAxis("x").contain(e[0]) && this.getAxis("y").contain(e[2]) && this.getAxis("z").contain(e[1]) }, containData: function(e) { return this.getAxis("x").containData(e[0]) && this.getAxis("y").containData(e[1]) && this.getAxis("z").containData(e[2]) }, dataToPoint: function(e, t, r) { return t = t || [], t[0] = this.getAxis("x").dataToCoord(e[0], r), t[2] = this.getAxis("y").dataToCoord(e[1], r), t[1] = this.getAxis("z").dataToCoord(e[2], r), t }, pointToData: function(e, t, r) { return t = t || [], t[0] = this.getAxis("x").coordToData(e[0], r), t[1] = this.getAxis("y").coordToData(e[2], r), t[2] = this.getAxis("z").coordToData(e[1], r), t } }, i.util.inherits(n, a), e.exports = n
    }, function(e, t, r) {
        function n(e, t) {
            var r = e.getBoxLayoutParams(),
                n = u.getLayoutRect(r, { width: t.getWidth(), height: t.getHeight() });
            n.y = t.getHeight() - n.y - n.height, this.viewGL.setViewport(n.x, n.y, n.width, n.height, t.getDevicePixelRatio());
            var i = e.get("boxWidth"),
                a = e.get("boxHeight"),
                o = e.get("boxDepth");
            this.getAxis("x").setExtent(-i / 2, i / 2), this.getAxis("y").setExtent(o / 2, -o / 2), this.getAxis("z").setExtent(-a / 2, a / 2), this.size = [i, a, o]
        }

        function i(e, t) {
            function r(e, t) { n[e] = n[e] || [1 / 0, -1 / 0], n[e][0] = Math.min(t[0], n[e][0]), n[e][1] = Math.max(t[1], n[e][1]) }
            var n = {};
            e.eachSeries(function(e) {
                if (e.coordinateSystem === this) {
                    var t = e.getData();
                    ["x", "y", "z"].forEach(function(n) { r(n, t.getDataExtent(e.coordDimToDataDim(n)[0], !0)) })
                }
            }, this), ["xAxis3D", "yAxis3D", "zAxis3D"].forEach(function(t) {
                e.eachComponent(t, function(e) {
                    var r = t.charAt(0),
                        i = e.getReferringComponents("grid3D")[0],
                        a = i.coordinateSystem;
                    if (a === this) {
                        var u = a.getAxis(r);
                        if (!u) {
                            var h = s.helper.createScale(n[r] || [1 / 0, -1 / 0], e);
                            u = new o(r, h), u.type = e.get("type");
                            var l = "category" === u.type;
                            u.onBand = l && e.get("boundaryGap"), u.inverse = e.get("inverse"), e.axis = u, u.model = e, a.addAxis(u)
                        }
                    }
                }, this)
            }, this), this.resize(this.model, t)
        }
        var a = r(151),
            o = r(150),
            s = r(0),
            u = r(42),
            h = r(21),
            l = (r(4), {
                dimensions: a.prototype.dimensions,
                create: function(e, t) {
                    function r(e, t) { return s.map(function(r) { var n = e.getReferringComponents(r)[0]; return null == n && (n = t.getComponent(r)), n }) }
                    var o = [];
                    e.eachComponent("grid3D", function(e) {
                        e.__viewGL = e.__viewGL || new h;
                        var t = new a;
                        t.model = e, t.viewGL = e.__viewGL, e.coordinateSystem = t, o.push(t), t.resize = n, t.update = i
                    });
                    var s = ["xAxis3D", "yAxis3D", "zAxis3D"];
                    return e.eachSeries(function(t) {
                        if ("cartesian3D" === t.get("coordinateSystem")) {
                            var n = t.getReferringComponents("grid3D")[0];
                            if (null == n) {
                                var i = r(t, e),
                                    n = i[0].getCoordSysModel();
                                i.forEach(function(e) { e.getCoordSysModel() })
                            }
                            var a = n.coordinateSystem;
                            t.coordinateSystem = a
                        }
                    }), o
                }
            });
        s.registerCoordinateSystem("grid3D", l), e.exports = l
    }, function(e, t, r) {
        function n() { this.width = 0, this.height = 0, this.altitudeScale = 1, this.boxHeight = "auto", this.altitudeExtent, this.bearing = 0, this.pitch = 0, this.center = [0, 0], this._origin, this.zoom = 0, this._initialZoom }
        var i = (r(0), r(1)),
            a = (r(3), r(9), i.vec3, i.mat4),
            o = .6435011087932844,
            s = Math.PI;
        n.prototype = {
            constructor: n,
            type: "mapbox",
            dimensions: ["lng", "lat", "alt"],
            containPoint: function() {},
            setCameraOption: function(e) { this.bearing = e.bearing, this.pitch = e.pitch, this.center = e.center, this.zoom = e.zoom, this._origin || (this._origin = this.projectOnTileWithScale(this.center, 512)), null == this._initialZoom && (this._initialZoom = this.zoom), this.updateTransform() },
            updateTransform: function() {
                if (this.height) {
                    var e = .5 / Math.tan(o / 2) * this.height * .1,
                        t = Math.max(Math.min(this.pitch, 60), 0) / 180 * Math.PI,
                        r = Math.PI / 2 + t,
                        n = Math.sin(o / 2) * e / Math.sin(Math.PI - r - o / 2),
                        i = Math.cos(Math.PI / 2 - t) * n + e,
                        s = 1.1 * i,
                        u = new Float64Array(16);
                    a.perspective(u, o, this.width / this.height, 1, s), this.viewGL.camera.projectionMatrix.setArray(u), this.viewGL.camera.decomposeProjectionMatrix();
                    var u = a.identity(new Float64Array(16)),
                        h = this.dataToPoint(this.center);
                    a.scale(u, u, [1, -1, 1]), a.translate(u, u, [0, 0, -e]), a.rotateX(u, u, t), a.rotateZ(u, u, -this.bearing / 180 * Math.PI), a.translate(u, u, [-h[0] * this.getScale() * .1, -h[1] * this.getScale() * .1, 0]), this.viewGL.camera.viewMatrix._array = u;
                    var l = new Float64Array(16);
                    a.invert(l, u), this.viewGL.camera.worldTransform._array = l, this.viewGL.camera.decomposeWorldTransform();
                    var c, d = 512 * this.getScale();
                    if (this.altitudeExtent && !isNaN(this.boxHeight)) {
                        var f = this.altitudeExtent[1] - this.altitudeExtent[0];
                        c = this.boxHeight / f * this.getScale() / Math.pow(2, this._initialZoom)
                    } else c = d / (2 * Math.PI * 6378e3 * Math.abs(Math.cos(this.center[1] * (Math.PI / 180)))) * this.altitudeScale * .1;
                    this.viewGL.rootNode.scale.set(.1 * this.getScale(), .1 * this.getScale(), c)
                }
            },
            getScale: function() { return Math.pow(2, this.zoom) },
            projectOnTile: function(e, t) { return this.projectOnTileWithScale(e, 512 * this.getScale(), t) },
            projectOnTileWithScale: function(e, t, r) {
                var n = e[0],
                    i = e[1],
                    a = n * s / 180,
                    o = i * s / 180,
                    u = t * (a + s) / (2 * s),
                    h = t * (s - Math.log(Math.tan(s / 4 + .5 * o))) / (2 * s);
                return r = r || [], r[0] = u, r[1] = h, r
            },
            unprojectFromTile: function(e, t) { return this.unprojectOnTileWithScale(e, 512 * this.getScale(), t) },
            unprojectOnTileWithScale: function(e, t, r) {
                var n = e[0],
                    i = e[1],
                    a = n / t * (2 * s) - s,
                    o = 2 * (Math.atan(Math.exp(s - i / t * (2 * s))) - s / 4);
                return r = r || [], r[0] = 180 * a / s, r[1] = 180 * o / s, r
            },
            dataToPoint: function(e, t) { return t = this.projectOnTileWithScale(e, 512, t), t[0] -= this._origin[0], t[1] -= this._origin[1], t[2] = isNaN(e[2]) ? 0 : e[2], isNaN(e[2]) || (t[2] = e[2], this.altitudeExtent && (t[2] -= this.altitudeExtent[0])), t }
        }, e.exports = n
    }, function(e, t, r) {
        function n(e, t) {
            var r = t.getWidth(),
                n = t.getHeight(),
                i = t.getDevicePixelRatio();
            this.viewGL.setViewport(0, 0, r, n, i), this.width = r, this.height = n, this.altitudeScale = e.get("altitudeScale"), this.boxHeight = e.get("boxHeight")
        }

        function i(e, t) {
            if ("auto" !== this.model.get("boxHeight")) {
                var r = [1 / 0, -1 / 0];
                e.eachSeries(function(e) {
                    if (e.coordinateSystem === this) {
                        var t = e.getData(),
                            n = e.coordDimToDataDim("alt")[0];
                        if (n) {
                            var i = t.getDataExtent(n, !0);
                            r[0] = Math.min(r[0], i[0]), r[1] = Math.max(r[1], i[1])
                        }
                    }
                }, this), r && isFinite(r[1] - r[0]) && (this.altitudeExtent = r)
            }
        }
        var a = r(153),
            o = r(0),
            s = r(4),
            u = r(2),
            h = r(21),
            l = {
                dimensions: a.prototype.dimensions,
                create: function(e, t) {
                    var r = [];
                    return e.eachComponent("mapbox", function(e) {
                        var o = e.__viewGL;
                        o || (o = e.__viewGL = new h, o.setRootNode(new u.Node));
                        var s = new a;
                        s.viewGL = e.__viewGL, s.resize = n, s.resize(e, t), r.push(s), e.coordinateSystem = s, s.model = e, s.setCameraOption(e.getMapboxCameraOption()), s.update = i
                    }), e.eachSeries(function(t) {
                        if ("mapbox" === t.get("coordinateSystem")) {
                            var r = t.getReferringComponents("mapbox")[0];
                            if (r || (r = e.getComponent("mapbox")), !r) throw new Error('mapbox "' + s.firstNotNull(t.get("mapboxIndex"), t.get("mapboxId"), 0) + '" not found');
                            t.coordinateSystem = r.coordinateSystem
                        }
                    }), r
                }
            };
        o.registerCoordinateSystem("mapbox", l), e.exports = l
    }, function(e, t, r) {
        function n(e) {
            var t = e.__zr;
            e.__zr = null, t && e.removeAnimatorsFromZr && e.removeAnimatorsFromZr(t)
        }

        function i(e) { return e.__GUID__ }

        function a(e, t, r) {
            var n = 0,
                i = [];
            for (var a in t) t[a].count ? n++ : i.push(t[a].target);
            for (var o = 0; o < Math.min(n - r, i.length); o++) i[o].dispose(e)
        }

        function o(e, t) {
            var r = i(t);
            e[r] = e[r] || { count: 0, target: t }, e[r].count++
        }
        var s = r(0),
            u = r(52),
            h = r(210),
            l = r(6),
            c = r(53),
            d = r(83),
            f = function(e, t) {
                this.id = e, this.zr = t;
                try { this.renderer = new u({ clearBit: 0, devicePixelRatio: t.painter.dpr, preserveDrawingBuffer: !0, premultipliedAlpha: !0 }), this.renderer.resize(t.painter.getWidth(), t.painter.getHeight()) } catch (e) { return this.renderer = null, this.dom = document.createElement("div"), this.dom.style.cssText = "position:absolute; left: 0; top: 0; right: 0; bottom: 0;", this.dom.className = "ecgl-nowebgl", this.dom.innerHTML = "Sorry, your browser does support WebGL", void console.error(e) }
                this.onglobalout = this.onglobalout.bind(this), t.on("globalout", this.onglobalout), this.dom = this.renderer.canvas;
                var r = this.dom.style;
                r.position = "absolute", r.left = "0", r.top = "0", this.views = [], this._picking = new h({ renderer: this.renderer }), this._viewsToDispose = [], this._accumulatingId = 0, this._zrEventProxy = new s.graphic.Rect({ shape: { x: -1, y: -1, width: 2, height: 2 }, __isGLToZRProxy: !0 })
            };
        f.prototype.addView = function(e) {
            if (e.layer !== this) {
                var t = this._viewsToDispose.indexOf(e);
                t >= 0 && this._viewsToDispose.splice(t, 1), this.views.push(e), e.layer = this;
                var r = this.zr;
                e.scene.traverse(function(e) { e.__zr = r, e.addAnimatorsToZr && e.addAnimatorsToZr(r) })
            }
        }, f.prototype.removeView = function(e) {
            if (e.layer === this) {
                var t = this.views.indexOf(e);
                t >= 0 && (this.views.splice(t, 1), e.scene.traverse(n, this), e.layer = null, this._viewsToDispose.push(e))
            }
        }, f.prototype.removeViewsAll = function() { this.views.forEach(function(e) { e.scene.traverse(n, this), e.layer = null, this._viewsToDispose.push(e) }, this), this.views.length = 0 }, f.prototype.resize = function(e, t) { this.renderer.resize(e, t) }, f.prototype.clear = function() {
            var e = this.renderer.gl;
            e.clearColor(0, 0, 0, 0), e.depthMask(!0), e.colorMask(!0, !0, !0, !0), e.clear(e.DEPTH_BUFFER_BIT | e.COLOR_BUFFER_BIT)
        }, f.prototype.clearDepth = function() {
            var e = this.renderer.gl;
            e.clear(e.DEPTH_BUFFER_BIT)
        }, f.prototype.clearColor = function() {
            var e = this.renderer.gl;
            e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT)
        }, f.prototype.needsRefresh = function() { this.zr.refresh() }, f.prototype.refresh = function() {
            for (var e = 0; e < this.views.length; e++) this.views[e].prepareRender();
            this._doRender(!1), this._trackAndClean();
            for (var e = 0; e < this._viewsToDispose.length; e++) this._viewsToDispose[e].dispose(this.renderer);
            this._viewsToDispose.length = 0, this._startAccumulating()
        }, f.prototype.renderToCanvas = function(e) { this._startAccumulating(!0), e.drawImage(this.dom, 0, 0, e.canvas.width, e.canvas.height) }, f.prototype._doRender = function(e) {
            this.clear(), this.renderer.saveViewport();
            for (var t = 0; t < this.views.length; t++) this.views[t].render(this.renderer, e);
            this.renderer.restoreViewport()
        }, f.prototype._stopAccumulating = function() { this._accumulatingId = 0, clearTimeout(this._accumulatingTimeout) };
        var p = 1;
        f.prototype._startAccumulating = function(e) {
            function t(i) {
                if (r._accumulatingId && i === r._accumulatingId) {
                    for (var a = !0, o = 0; o < r.views.length; o++) a = r.views[o].isAccumulateFinished() && n;
                    a || (r._doRender(!0), e ? t(i) : d(function() { t(i) }))
                }
            }
            var r = this;
            this._stopAccumulating();
            for (var n = !1, i = 0; i < this.views.length; i++) n = this.views[i].needsAccumulate() || n;
            n && (this._accumulatingId = p++, e ? t(r._accumulatingId) : this._accumulatingTimeout = setTimeout(function() { t(r._accumulatingId) }, 50))
        }, f.prototype._trackAndClean = function() {
            function e(e) {
                for (var i = 0; i < e.length; i++) {
                    var a = e[i],
                        s = a.geometry,
                        u = a.material,
                        h = u.shader;
                    o(n, s), o(t, h);
                    for (var c in u.uniforms) {
                        var d = u.uniforms[c].value;
                        if (d instanceof l) o(r, d);
                        else if (d instanceof Array)
                            for (var f = 0; f < d.length; f++) d[f] instanceof l && o(r, d[f])
                    }
                }
            }
            var t = this._shadersMap = this._shadersMap || {},
                r = this._texturesMap = this._texturesMap || {},
                n = this._geometriesMap = this._geometriesMap || {};
            for (var i in t) t[i].count = 0;
            for (var i in r) r[i].count = 0;
            for (var i in n) n[i].count = 0;
            for (var s = 0; s < this.views.length; s++) {
                var u = this.views[s],
                    h = u.scene;
                e(h.opaqueQueue), e(h.transparentQueue);
                for (var c = 0; c < h.lights.length; c++) h.lights[c].cubemap && o(r, h.lights[c].cubemap)
            }
            var d = this.renderer.gl;
            a(d, t, 60), a(d, r, 20), a(d, n, 20)
        }, f.prototype.dispose = function() { this._stopAccumulating(), this.renderer.disposeScene(this.scene), this.zr.off("globalout", this.onglobalout) }, f.prototype.onmousedown = function(e) {
            if (!e.target || !e.target.__isGLToZRProxy) {
                e = e.event;
                var t = this.pickObject(e.offsetX, e.offsetY);
                t && (this._dispatchEvent("mousedown", e, t), this._dispatchDataEvent("mousedown", e, t)), this._downX = e.offsetX, this._downY = e.offsetY
            }
        }, f.prototype.onmousemove = function(e) {
            if (!e.target || !e.target.__isGLToZRProxy) {
                e = e.event;
                var t = this.pickObject(e.offsetX, e.offsetY),
                    r = t && t.target,
                    n = this._hovered;
                this._hovered = t, n && r !== n.target && (n.relatedTarget = r, this._dispatchEvent("mouseout", e, n), this.zr.setCursorStyle("default")), this._dispatchEvent("mousemove", e, t), t && (this.zr.setCursorStyle("pointer"), n && r === n.target || this._dispatchEvent("mouseover", e, t)), this._dispatchDataEvent("mousemove", e, t)
            }
        }, f.prototype.onmouseup = function(e) {
            if (!e.target || !e.target.__isGLToZRProxy) {
                e = e.event;
                var t = this.pickObject(e.offsetX, e.offsetY);
                t && (this._dispatchEvent("mouseup", e, t), this._dispatchDataEvent("mouseup", e, t)), this._upX = e.offsetX, this._upY = e.offsetY
            }
        }, f.prototype.onclick = f.prototype.dblclick = function(e) {
            if (!e.target || !e.target.__isGLToZRProxy) {
                var t = this._upX - this._downX,
                    r = this._upY - this._downY;
                if (!(Math.sqrt(t * t + r * r) > 20)) {
                    e = e.event;
                    var n = this.pickObject(e.offsetX, e.offsetY);
                    n && (this._dispatchEvent(e.type, e, n), this._dispatchDataEvent(e.type, e, n));
                    var i = this._clickToSetFocusPoint(e);
                    if (i) { i.view.setDOFFocusOnPoint(i.distance) && this.zr.refresh() }
                }
            }
        }, f.prototype._clickToSetFocusPoint = function(e) {
            for (var t = this.renderer, r = t.viewport, n = this.views.length - 1; n >= 0; n--) { var i = this.views[n]; if (i.hasDOF() && i.containPoint(e.offsetX, e.offsetY)) { this._picking.scene = i.scene, this._picking.camera = i.camera, t.viewport = i.viewport; var a = this._picking.pick(e.offsetX, e.offsetY, !0); if (a) return a.view = i, a } }
            t.viewport = r
        }, f.prototype.onglobalout = function(e) {
            var t = this._hovered;
            t && this._dispatchEvent("mouseout", e, { target: t.target })
        }, f.prototype.pickObject = function(e, t) {
            for (var r = [], n = this.renderer, i = n.viewport, a = 0; a < this.views.length; a++) {
                var o = this.views[a];
                o.containPoint(e, t) && (this._picking.scene = o.scene, this._picking.camera = o.camera, n.viewport = o.viewport, this._picking.pickAll(e, t, r))
            }
            return n.viewport = i, r.sort(function(e, t) { return e.distance - t.distance }), r[0]
        }, f.prototype._dispatchEvent = function(e, t, r) {
            r || (r = {});
            var n = r.target;
            for (r.cancelBubble = !1, r.event = t, r.type = e, r.offsetX = t.offsetX, r.offsetY = t.offsetY; n && (n.trigger(e, r), n = n.getParent(), !r.cancelBubble););
            this._dispatchToView(e, r)
        }, f.prototype._dispatchDataEvent = function(e, t, r) {
            var n = r && r.target,
                i = n && n.dataIndex,
                a = n && n.seriesIndex,
                o = n && n.eventData,
                s = !1,
                u = this._zrEventProxy;
            u.position = [t.offsetX, t.offsetY], u.update();
            var h = { target: u };
            "mousemove" === e && (null != i ? i !== this._lastDataIndex && (parseInt(this._lastDataIndex, 10) >= 0 && (u.dataIndex = this._lastDataIndex, u.seriesIndex = this._lastSeriesIndex, this.zr.handler.dispatchToElement(h, "mouseout", t)), s = !0) : null != o && o !== this._lastEventData && (null != this._lastEventData && (u.eventData = this._lastEventData, this.zr.handler.dispatchToElement(h, "mouseout", t)), s = !0), this._lastEventData = o, this._lastDataIndex = i, this._lastSeriesIndex = a), u.eventData = o, u.dataIndex = i, u.seriesIndex = a, (null != o || parseInt(i, 10) >= 0) && (this.zr.handler.dispatchToElement(h, e, t), s && this.zr.handler.dispatchToElement(h, "mouseover", t))
        }, f.prototype._dispatchToView = function(e, t) { for (var r = 0; r < this.views.length; r++) this.views[r].containPoint(t.offsetX, t.offsetY) && this.views[r].trigger(e, t) }, s.util.extend(f.prototype, c), e.exports = f
    }, function(e, t) { e.exports = "@export ecgl.dof.coc\n\nuniform sampler2D depth;\n\nuniform float zNear: 0.1;\nuniform float zFar: 2000;\n\nuniform float focalDistance: 3;\nuniform float focalRange: 1;\nuniform float focalLength: 30;\nuniform float fstop: 2.8;\n\nvarying vec2 v_Texcoord;\n\n@import qtek.util.encode_float\n\nvoid main()\n{\n float z = texture2D(depth, v_Texcoord).r * 2.0 - 1.0;\n\n float dist = 2.0 * zNear * zFar / (zFar + zNear - z * (zFar - zNear));\n\n float aperture = focalLength / fstop;\n\n float coc;\n\n float uppper = focalDistance + focalRange;\n float lower = focalDistance - focalRange;\n if (dist <= uppper && dist >= lower) {\n coc = 0.5;\n }\n else {\n float focalAdjusted = dist > uppper ? uppper : lower;\n\n coc = abs(aperture * (focalLength * (dist - focalAdjusted)) / (dist * (focalAdjusted - focalLength)));\n coc = clamp(coc, 0.0, 0.4) / 0.4000001;\n\n if (dist < lower) {\n coc = -coc;\n }\n coc = coc * 0.5 + 0.5;\n }\n\n gl_FragColor = encodeFloat(coc);\n}\n@end\n\n\n@export ecgl.dof.composite\n\n#define DEBUG 0\n\nuniform sampler2D original;\nuniform sampler2D blurred;\nuniform sampler2D nearfield;\nuniform sampler2D coc;\nuniform sampler2D nearcoc;\nvarying vec2 v_Texcoord;\n\n@import qtek.util.rgbm\n@import qtek.util.float\n\nvoid main()\n{\n vec4 blurredColor = decodeHDR(texture2D(blurred, v_Texcoord));\n vec4 originalColor = decodeHDR(texture2D(original, v_Texcoord));\n\n float fCoc = decodeFloat(texture2D(coc, v_Texcoord));\n\n fCoc = abs(fCoc * 2.0 - 1.0);\n\n float weight = smoothstep(0.0, 1.0, fCoc);\n \n#ifdef NEARFIELD_ENABLED\n vec4 nearfieldColor = decodeHDR(texture2D(nearfield, v_Texcoord));\n float fNearCoc = decodeFloat(texture2D(nearcoc, v_Texcoord));\n fNearCoc = abs(fNearCoc * 2.0 - 1.0);\n\n gl_FragColor = encodeHDR(\n mix(\n nearfieldColor, mix(originalColor, blurredColor, weight),\n pow(1.0 - fNearCoc, 4.0)\n )\n );\n#else\n gl_FragColor = encodeHDR(mix(originalColor, blurredColor, weight));\n#endif\n\n}\n\n@end\n\n\n\n@export ecgl.dof.diskBlur\n\n#define POISSON_KERNEL_SIZE 16;\n\nuniform sampler2D texture;\nuniform sampler2D coc;\nvarying vec2 v_Texcoord;\n\nuniform float blurRadius : 10.0;\nuniform vec2 textureSize : [512.0, 512.0];\n\nuniform vec2 poissonKernel[POISSON_KERNEL_SIZE];\n\nuniform float percent;\n\nfloat nrand(const in vec2 n) {\n return fract(sin(dot(n.xy ,vec2(12.9898,78.233))) * 43758.5453);\n}\n\n@import qtek.util.rgbm\n@import qtek.util.float\n\n\nvoid main()\n{\n vec2 offset = blurRadius / textureSize;\n\n float rnd = 6.28318 * nrand(v_Texcoord + 0.07 * percent );\n float cosa = cos(rnd);\n float sina = sin(rnd);\n vec4 basis = vec4(cosa, -sina, sina, cosa);\n\n#if !defined(BLUR_NEARFIELD) && !defined(BLUR_COC)\n offset *= abs(decodeFloat(texture2D(coc, v_Texcoord)) * 2.0 - 1.0);\n#endif\n\n#ifdef BLUR_COC\n float cocSum = 0.0;\n#else\n vec4 color = vec4(0.0);\n#endif\n\n\n float weightSum = 0.0;\n\n for (int i = 0; i < POISSON_KERNEL_SIZE; i++) {\n vec2 ofs = poissonKernel[i];\n\n ofs = vec2(dot(ofs, basis.xy), dot(ofs, basis.zw));\n\n vec2 uv = v_Texcoord + ofs * offset;\n vec4 texel = texture2D(texture, uv);\n\n float w = 1.0;\n#ifdef BLUR_COC\n float fCoc = decodeFloat(texel) * 2.0 - 1.0;\n cocSum += clamp(fCoc, -1.0, 0.0) * w;\n#else\n texel = decodeHDR(texel);\n #if !defined(BLUR_NEARFIELD)\n float fCoc = decodeFloat(texture2D(coc, uv)) * 2.0 - 1.0;\n w *= abs(fCoc);\n #endif\n color += texel * w;\n#endif\n\n weightSum += w;\n }\n\n#ifdef BLUR_COC\n gl_FragColor = encodeFloat(clamp(cocSum / weightSum, -1.0, 0.0) * 0.5 + 0.5);\n#else\n color /= weightSum;\n gl_FragColor = encodeHDR(color);\n#endif\n}\n\n@end" }, function(e, t, r) {
        function n(e) { e = e || {}, this._edgePass = new o({ fragment: s.source("ecgl.edge") }), this._edgePass.setUniform("normalTexture", e.normalTexture), this._edgePass.setUniform("depthTexture", e.depthTexture), this._targetTexture = new i({ type: a.HALF_FLOAT }), this._frameBuffer = new u, this._frameBuffer.attach(this._targetTexture) }
        var i = (r(9), r(3), r(5)),
            a = r(6),
            o = r(12),
            s = r(7),
            u = r(10);
        n.prototype.update = function(e, t, r, n) {
            var i = e.getWidth(),
                a = e.getHeight(),
                o = this._targetTexture;
            o.width = i, o.height = a;
            var s = this._frameBuffer;
            s.bind(e), this._edgePass.setUniform("projectionInv", t.invProjectionMatrix._array), this._edgePass.setUniform("textureSize", [i, a]), this._edgePass.setUniform("texture", r), this._edgePass.render(e), s.unbind(e)
        }, n.prototype.getTargetTexture = function() { return this._targetTexture }, n.prototype.setParameter = function(e, t) { this._edgePass.setUniform(e, t) }, n.prototype.dispose = function(e) { this._targetTexture.dispose(e), this._frameBuffer.dispose(e) }, e.exports = n
    }, function(e, t, r) {
        function n() {
            this._sourceTexture = new a({ type: o.HALF_FLOAT }), this._depthTexture = new a({ format: o.DEPTH_COMPONENT, type: o.UNSIGNED_INT }), this._framebuffer = new s, this._framebuffer.attach(this._sourceTexture), this._framebuffer.attach(this._depthTexture, s.DEPTH_ATTACHMENT), this._normalPass = new f;
            var e = new u;
            this._compositor = e.parse(_);
            var t = this._compositor.getNodeByName("source");
            t.texture = this._sourceTexture;
            var r = this._compositor.getNodeByName("coc");
            this._sourceNode = t, this._cocNode = r, this._compositeNode = this._compositor.getNodeByName("composite"), this._fxaaNode = this._compositor.getNodeByName("FXAA"), this._dofBlurNodes = ["dof_far_blur", "dof_near_blur", "dof_coc_blur"].map(function(e) { return this._compositor.getNodeByName(e) }, this), this._dofBlurKernel = 0, this._dofBlurKernelSize = new Float32Array(0), this._finalNodesChain = g.map(function(e) { return this._compositor.getNodeByName(e) }, this);
            var n = { normalTexture: this._normalPass.getNormalTexture(), depthTexture: this._normalPass.getDepthTexture() };
            this._ssaoPass = new h(n), this._ssrPass = new l(n), this._edgePass = new p(n)
        }
        var i = (r(71), r(7)),
            a = r(5),
            o = r(6),
            s = r(10),
            u = r(205),
            h = r(161),
            l = r(163),
            c = r(167),
            d = r(2),
            f = r(159),
            p = r(157),
            _ = (r(9), r(165));
        i.import(r(214)), i.import(r(219)), i.import(r(220)), i.import(r(215)), i.import(r(216)), i.import(r(221)), i.import(r(218)), i.import(r(213)), i.import(r(217)), i.import(r(156)), i.import(r(166));
        var m = { color: { parameters: { width: function(e) { return e.getWidth() }, height: function(e) { return e.getHeight() } } } },
            g = ["composite", "FXAA"];
        n.prototype.resize = function(e, t, r) {
            r = r || 1;
            var e = e * r,
                t = t * r,
                n = this._sourceTexture,
                i = this._depthTexture;
            n.width = e, n.height = t, i.width = e, i.height = t
        }, n.prototype._ifRenderNormalPass = function() { return this._enableSSAO || this._enableEdge || this._enableSSR }, n.prototype._getPrevNode = function(e) { for (var t = g.indexOf(e.name) - 1, r = this._finalNodesChain[t]; r && !this._compositor.getNodeByName(r.name);) t -= 1, r = this._finalNodesChain[t]; return r }, n.prototype._getNextNode = function(e) { for (var t = g.indexOf(e.name) + 1, r = this._finalNodesChain[t]; r && !this._compositor.getNodeByName(r.name);) t += 1, r = this._finalNodesChain[t]; return r }, n.prototype._addChainNode = function(e) {
            var t = this._getPrevNode(e),
                r = this._getNextNode(e);
            t && (t.outputs = m, e.inputs.texture = t.name, r ? (e.outputs = m, r.inputs.texture = e.name) : e.outputs = null, this._compositor.addNode(e))
        }, n.prototype._removeChainNode = function(e) {
            var t = this._getPrevNode(e),
                r = this._getNextNode(e);
            t && (r ? (t.outputs = m, r.inputs.texture = t.name) : t.outputs = null, this._compositor.removeNode(e))
        }, n.prototype.updateNormal = function(e, t, r, n) { this._ifRenderNormalPass() && this._normalPass.update(e, t, r) }, n.prototype.updateSSAO = function(e, t, r, n) { this._ssaoPass.update(e, r, n) }, n.prototype.enableSSAO = function() { this._enableSSAO = !0 }, n.prototype.disableSSAO = function() { this._enableSSAO = !1 }, n.prototype.enableSSR = function() { this._enableSSR = !0 }, n.prototype.disableSSR = function() { this._enableSSR = !1 }, n.prototype.getSSAOTexture = function(e, t, r, n) { return this._ssaoPass.getTargetTexture() }, n.prototype.getSourceFrameBuffer = function() { return this._framebuffer }, n.prototype.getSourceTexture = function() { return this._sourceTexture }, n.prototype.disableFXAA = function() { this._removeChainNode(this._fxaaNode) }, n.prototype.enableFXAA = function() { this._addChainNode(this._fxaaNode) }, n.prototype.enableBloom = function() { this._compositeNode.inputs.bloom = "bloom_composite" }, n.prototype.disableBloom = function() { this._compositeNode.inputs.bloom = null }, n.prototype.enableDOF = function() { this._compositeNode.inputs.texture = "dof_composite" }, n.prototype.disableDOF = function() { this._compositeNode.inputs.texture = "source" }, n.prototype.enableColorCorrection = function() { this._compositeNode.shaderDefine("COLOR_CORRECTION"), this._enableColorCorrection = !0 }, n.prototype.disableColorCorrection = function() { this._compositeNode.shaderUndefine("COLOR_CORRECTION"), this._enableColorCorrection = !1 }, n.prototype.enableEdge = function() { this._enableEdge = !0 }, n.prototype.disableEdge = function() { this._enableEdge = !1 }, n.prototype.setBloomIntensity = function(e) { this._compositeNode.setParameter("bloomIntensity", e) }, n.prototype.setSSAOParameter = function(e, t) {
            switch (e) {
                case "quality":
                    var r = { low: 6, medium: 12, high: 32, ultra: 62 }[t] || 12;
                    this._ssaoPass.setParameter("kernelSize", r);
                    break;
                case "radius":
                    this._ssaoPass.setParameter(e, t), this._ssaoPass.setParameter("bias", t / 200);
                    break;
                case "intensity":
                    this._ssaoPass.setParameter(e, t)
            }
        }, n.prototype.setDOFParameter = function(e, t) {
            switch (e) {
                case "focalDistance":
                case "focalRange":
                case "fstop":
                    this._cocNode.setParameter(e, t);
                    break;
                case "blurRadius":
                    for (var r = 0; r < this._dofBlurNodes.length; r++) this._dofBlurNodes[r].setParameter("blurRadius", t);
                    break;
                case "quality":
                    var n = { low: 4, medium: 8, high: 16, ultra: 32 }[t] || 8;
                    this._dofBlurKernelSize = n;
                    for (var r = 0; r < this._dofBlurNodes.length; r++) this._dofBlurNodes[r].shaderDefine("POISSON_KERNEL_SIZE", n);
                    this._dofBlurKernel = new Float32Array(2 * n)
            }
        }, n.prototype.setSSRParameter = function(e, t) {
            switch (e) {
                case "quality":
                    var r = { low: 10, medium: 20, high: 40, ultra: 80 }[t] || 20,
                        n = { low: 32, medium: 16, high: 8, ultra: 4 }[t] || 16;
                    this._ssrPass.setParameter("maxIteration", r), this._ssrPass.setParameter("pixelStride", n);
                    break;
                case "maxRoughness":
                    this._ssrPass.setParameter("minGlossiness", Math.max(Math.min(1 - t, 1), 0))
            }
        }, n.prototype.setEdgeColor = function(e) {
            var t = d.parseColor(e);
            this._edgePass.setParameter("edgeColor", t)
        }, n.prototype.setExposure = function(e) { this._compositeNode.setParameter("exposure", Math.pow(2, e)) }, n.prototype.setColorLookupTexture = function(e, t) { this._compositeNode.pass.material.setTextureImage("lut", this._enableColorCorrection ? e : "none", t, { minFilter: d.Texture.NEAREST, magFilter: d.Texture.NEAREST, flipY: !1 }) }, n.prototype.setColorCorrection = function(e, t) { this._compositeNode.setParameter(e, t) }, n.prototype.composite = function(e, t, r, n) {
            var i = this._sourceTexture,
                a = i;
            this._enableEdge && (this._edgePass.update(e, t, i, n), i = a = this._edgePass.getTargetTexture()), this._enableSSR && (this._ssrPass.update(e, t, i, n), a = this._ssrPass.getTargetTexture()), this._sourceNode.texture = a, this._cocNode.setParameter("depth", this._depthTexture);
            for (var o = this._dofBlurKernel, s = this._dofBlurKernelSize, u = Math.floor(c.length / 2 / s), h = n % u, l = 0; l < 2 * s; l++) o[l] = c[l + h * s * 2];
            for (var l = 0; l < this._dofBlurNodes.length; l++) this._dofBlurNodes[l].setParameter("percent", n / 30), this._dofBlurNodes[l].setParameter("poissonKernel", o);
            this._cocNode.setParameter("zNear", t.near), this._cocNode.setParameter("zFar", t.far), this._compositor.render(e, r)
        }, n.prototype.dispose = function(e) { this._sourceTexture.dispose(e), this._depthTexture.dispose(e), this._framebuffer.dispose(e), this._compositor.dispose(e), this._normalPass.dispose(e), this._ssaoPass.dispose(e) }, e.exports = n
    }, function(e, t, r) {
        function n(e, t, r, n, i) { t.setUniform(e, "1i", r, i), e.activeTexture(e.TEXTURE0 + i), n.isRenderable() ? n.bind(e) : n.unbind(e) }

        function i(e, t, r, i, a) {
            var o, s, u, h;
            return function(l, c, d) {
                if (!h || h.material !== l.material) {
                    var f = l.material,
                        p = f.get("roughness");
                    null == p && (p = 1);
                    var _ = f.get("normalMap") || t,
                        m = f.get("roughnessMap"),
                        g = f.get("bumpMap"),
                        v = f.get("uvRepeat"),
                        y = f.get("uvOffset"),
                        x = f.get("detailUvRepeat"),
                        T = f.get("detailUvOffset"),
                        b = !!g && f.shader.isTextureEnabled("bumpMap"),
                        w = !!m && f.shader.isTextureEnabled("roughnessMap"),
                        E = f.shader.isDefined("fragment", "DOUBLE_SIDED");
                    g = g || r, m = m || i, c !== a ? (a.set("normalMap", _), a.set("bumpMap", g), a.set("roughnessMap", m), a.set("useBumpMap", b), a.set("useRoughnessMap", w), a.set("doubleSide", E), null != v && a.set("uvRepeat", v), null != y && a.set("uvOffset", y), null != x && a.set("detailUvRepeat", x), null != T && a.set("detailUvOffset", T), a.set("roughness", p)) : (a.shader.setUniform(e, "1f", "roughness", p), o !== _ && n(e, a.shader, "normalMap", _, 0), s !== g && g && n(e, a.shader, "bumpMap", g, 1), u !== m && m && n(e, a.shader, "roughnessMap", m, 2), null != v && a.shader.setUniform(e, "2f", "uvRepeat", v), null != y && a.shader.setUniform(e, "2f", "uvOffset", y), null != x && a.shader.setUniform(e, "2f", "detailUvRepeat", x), null != T && a.shader.setUniform(e, "2f", "detailUvOffset", T), a.shader.setUniform(e, "1i", "useBumpMap", +b), a.shader.setUniform(e, "1i", "useRoughnessMap", +w), a.shader.setUniform(e, "1i", "doubleSide", +E)), o = _, s = g, u = m, h = l
                }
            }
        }

        function a(e) { e = e || {}, this._depthTex = new o({ format: s.DEPTH_COMPONENT, type: s.UNSIGNED_INT }), this._normalTex = new o({ type: s.HALF_FLOAT }), this._framebuffer = new h, this._framebuffer.attach(this._normalTex), this._framebuffer.attach(this._depthTex, h.DEPTH_ATTACHMENT), this._normalMaterial = new l({ shader: new u({ vertex: u.source("ecgl.normal.vertex"), fragment: u.source("ecgl.normal.fragment") }) }), this._normalMaterial.shader.enableTexture(["normalMap", "bumpMap", "roughnessMap"]), this._defaultNormalMap = d.createBlank("#000"), this._defaultBumpMap = d.createBlank("#000"), this._defaultRoughessMap = d.createBlank("#000"), this._debugPass = new c({ fragment: u.source("qtek.compositor.output") }), this._debugPass.setUniform("texture", this._normalTex), this._debugPass.material.shader.undefine("fragment", "OUTPUT_ALPHA") }
        var o = r(5),
            s = r(6),
            u = r(7),
            h = r(10),
            l = r(16),
            u = r(7),
            c = r(12),
            d = r(47);
        u.import(r(185)), a.prototype.getDepthTexture = function() { return this._depthTex }, a.prototype.getNormalTexture = function() { return this._normalTex }, a.prototype.update = function(e, t, r) {
            var n = e.getWidth(),
                a = e.getHeight(),
                o = this._depthTex,
                s = this._normalTex;
            o.width = n, o.height = a, s.width = n, s.height = a;
            var u = t.opaqueQueue,
                h = e.ifRenderObject,
                l = e.beforeRenderObject;
            e.ifRenderObject = function(e) { return e.renderNormal }, e.beforeRenderObject = i(e.gl, this._defaultNormalMap, this._defaultBumpMap, this._defaultRoughessMap, this._normalMaterial), this._framebuffer.bind(e), e.gl.clearColor(0, 0, 0, 0), e.gl.clear(e.gl.COLOR_BUFFER_BIT | e.gl.DEPTH_BUFFER_BIT), e.gl.disable(e.gl.BLEND), e.renderQueue(u, r, this._normalMaterial), this._framebuffer.unbind(e), e.ifRenderObject = h, e.beforeRenderObject = l
        }, a.prototype.renderDebug = function(e) { this._debugPass.render(e) }, a.prototype.dispose = function(e) { this._depthTex.dispose(e), this._normalTex.dispose(e) }, e.exports = a
    }, function(e, t) { e.exports = "@export ecgl.ssao.estimate\n\nuniform sampler2D depthTex;\n\nuniform sampler2D normalTex;\n\nuniform sampler2D noiseTex;\n\nuniform vec2 depthTexSize;\n\nuniform vec2 noiseTexSize;\n\nuniform mat4 projection;\n\nuniform mat4 projectionInv;\n\nuniform mat4 viewInverseTranspose;\n\nuniform vec3 kernel[KERNEL_SIZE];\n\nuniform float radius : 1;\n\nuniform float power : 1;\n\nuniform float bias: 1e-2;\n\nuniform float intensity: 1.0;\n\nvarying vec2 v_Texcoord;\n\nfloat ssaoEstimator(in vec3 originPos, in mat3 kernelBasis) {\n float occlusion = 0.0;\n\n for (int i = 0; i < KERNEL_SIZE; i++) {\n vec3 samplePos = kernel[i];\n#ifdef NORMALTEX_ENABLED\n samplePos = kernelBasis * samplePos;\n#endif\n samplePos = samplePos * radius + originPos;\n\n vec4 texCoord = projection * vec4(samplePos, 1.0);\n texCoord.xy /= texCoord.w;\n\n vec4 depthTexel = texture2D(depthTex, texCoord.xy * 0.5 + 0.5);\n\n float sampleDepth = depthTexel.r * 2.0 - 1.0;\n if (projection[3][3] == 0.0) {\n sampleDepth = projection[3][2] / (sampleDepth * projection[2][3] - projection[2][2]);\n }\n else {\n sampleDepth = (sampleDepth - projection[3][2]) / projection[2][2];\n }\n \n float rangeCheck = smoothstep(0.0, 1.0, radius / abs(originPos.z - sampleDepth));\n occlusion += rangeCheck * step(samplePos.z, sampleDepth - bias);\n }\n#ifdef NORMALTEX_ENABLED\n occlusion = 1.0 - occlusion / float(KERNEL_SIZE);\n#else\n occlusion = 1.0 - clamp((occlusion / float(KERNEL_SIZE) - 0.6) * 2.5, 0.0, 1.0);\n#endif\n return pow(occlusion, power);\n}\n\nvoid main()\n{\n\n vec4 depthTexel = texture2D(depthTex, v_Texcoord);\n\n#ifdef NORMALTEX_ENABLED\n vec4 tex = texture2D(normalTex, v_Texcoord);\n if (dot(tex.rgb, tex.rgb) == 0.0) {\n gl_FragColor = vec4(1.0);\n return;\n }\n vec3 N = tex.rgb * 2.0 - 1.0;\n N = (viewInverseTranspose * vec4(N, 0.0)).xyz;\n\n vec2 noiseTexCoord = depthTexSize / vec2(noiseTexSize) * v_Texcoord;\n vec3 rvec = texture2D(noiseTex, noiseTexCoord).rgb * 2.0 - 1.0;\n vec3 T = normalize(rvec - N * dot(rvec, N));\n vec3 BT = normalize(cross(N, T));\n mat3 kernelBasis = mat3(T, BT, N);\n#else\n if (depthTexel.r > 0.99999) {\n gl_FragColor = vec4(1.0);\n return;\n }\n mat3 kernelBasis;\n#endif\n\n float z = depthTexel.r * 2.0 - 1.0;\n\n vec4 projectedPos = vec4(v_Texcoord * 2.0 - 1.0, z, 1.0);\n vec4 p4 = projectionInv * projectedPos;\n\n vec3 position = p4.xyz / p4.w;\n\n float ao = ssaoEstimator(position, kernelBasis);\n ao = clamp(1.0 - (1.0 - ao) * intensity, 0.0, 1.0);\n gl_FragColor = vec4(vec3(ao), 1.0);\n}\n\n@end\n\n\n@export ecgl.ssao.blur\n\nuniform sampler2D ssaoTexture;\n\nuniform vec2 textureSize;\n\nvarying vec2 v_Texcoord;\n\nvoid main ()\n{\n\n vec2 texelSize = 1.0 / textureSize;\n\n float ao = 0.0;\n vec2 hlim = vec2(float(-BLUR_SIZE) * 0.5 + 0.5);\n float centerAo = texture2D(ssaoTexture, v_Texcoord).r;\n float weightAll = 0.0;\n float boxWeight = 1.0 / float(BLUR_SIZE) * float(BLUR_SIZE);\n for (int x = 0; x < BLUR_SIZE; x++) {\n for (int y = 0; y < BLUR_SIZE; y++) {\n vec2 coord = (vec2(float(x), float(y)) + hlim) * texelSize + v_Texcoord;\n float sampleAo = texture2D(ssaoTexture, coord).r;\n float closeness = 1.0 - distance(sampleAo, centerAo) / sqrt(3.0);\n float weight = boxWeight * closeness;\n ao += weight * sampleAo;\n weightAll += weight;\n }\n }\n\n gl_FragColor = vec4(vec3(clamp(ao / weightAll, 0.0, 1.0)), 1.0);\n}\n@end" }, function(e, t, r) {
        function n(e) {
            for (var t = new Uint8Array(e * e * 4), r = 0, n = new u, i = 0; i < e; i++)
                for (var a = 0; a < e; a++) n.set(2 * Math.random() - 1, 2 * Math.random() - 1, 0).normalize(), t[r++] = 255 * (.5 * n.x + .5), t[r++] = 255 * (.5 * n.y + .5), t[r++] = 0, t[r++] = 255;
            return t
        }

        function i(e) { return new h({ pixels: n(e), wrapS: l.REPEAT, wrapT: l.REPEAT, width: e, height: e }) }

        function a(e, t, r) {
            var n = new Float32Array(3 * e);
            t = t || 0;
            for (var i = 0; i < e; i++) {
                var a = p(i + t, 2) * (r ? 1 : 2) * Math.PI,
                    o = p(i + t, 3) * Math.PI,
                    s = Math.random(),
                    u = Math.cos(a) * Math.sin(o) * s,
                    h = Math.cos(o) * s,
                    l = Math.sin(a) * Math.sin(o) * s;
                n[3 * i] = u, n[3 * i + 1] = h, n[3 * i + 2] = l
            }
            return n
        }

        function o(e) { e = e || {}, this._ssaoPass = new c({ fragment: d.source("ecgl.ssao.estimate") }), this._blurPass = new c({ fragment: d.source("ecgl.ssao.blur") }), this._framebuffer = new f, this._ssaoTexture = new h, this._targetTexture = new h, this._depthTex = e.depthTexture, this._normalTex = e.normalTexture, this.setNoiseSize(4), this.setKernelSize(e.kernelSize || 12), this.setParameter("blurSize", Math.round(e.blurSize || 4)), null != e.radius && this.setParameter("radius", e.radius), null != e.power && this.setParameter("power", e.power), this._normalTex || this._ssaoPass.material.shader.disableTexture("normalTex") }
        var s = r(9),
            u = r(3),
            h = r(5),
            l = r(6),
            c = r(12),
            d = r(7),
            f = r(10),
            p = r(39);
        d.import(r(160)), o.prototype.setDepthTexture = function(e) { this._depthTex = e }, o.prototype.setNormalTexture = function(e) { this._normalTex = e, this._ssaoPass.material.shader[e ? "enableTexture" : "disableTexture"]("normalTex"), this.setKernelSize(this._kernelSize) }, o.prototype.update = function(e, t, r) {
            var n = e.getWidth(),
                i = e.getHeight(),
                a = this._ssaoPass,
                o = this._blurPass;
            a.setUniform("kernel", this._kernels[r % this._kernels.length]), a.setUniform("depthTex", this._depthTex), null != this._normalTex && a.setUniform("normalTex", this._normalTex), a.setUniform("depthTexSize", [this._depthTex.width, this._depthTex.height]);
            var u = new s;
            s.transpose(u, t.worldTransform), a.setUniform("projection", t.projectionMatrix._array), a.setUniform("projectionInv", t.invProjectionMatrix._array), a.setUniform("viewInverseTranspose", u._array);
            var h = this._ssaoTexture,
                l = this._targetTexture;
            h.width = n, h.height = i, l.width = n, l.height = i, this._framebuffer.attach(h), this._framebuffer.bind(e), e.gl.clearColor(1, 1, 1, 1), e.gl.clear(e.gl.COLOR_BUFFER_BIT), a.render(e), this._framebuffer.attach(l), o.setUniform("textureSize", [n, i]), o.setUniform("ssaoTexture", this._ssaoTexture), o.render(e), this._framebuffer.unbind(e);
            var c = e.clearColor;
            e.gl.clearColor(c[0], c[1], c[2], c[3])
        }, o.prototype.getTargetTexture = function() { return this._targetTexture }, o.prototype.setParameter = function(e, t) { "noiseTexSize" === e ? this.setNoiseSize(t) : "kernelSize" === e ? this.setKernelSize(t) : "blurSize" === e ? this._blurPass.material.shader.define("fragment", "BLUR_SIZE", t) : "intensity" === e ? this._ssaoPass.material.set("intensity", t) : this._ssaoPass.setUniform(e, t) }, o.prototype.setKernelSize = function(e) { this._kernelSize = e, this._ssaoPass.material.shader.define("fragment", "KERNEL_SIZE", e), this._kernels = this._kernels || []; for (var t = 0; t < 30; t++) this._kernels[t] = a(e, t * e, !!this._normalTex) }, o.prototype.setNoiseSize = function(e) {
            var t = this._ssaoPass.getUniform("noiseTex");
            t ? (t.data = n(e), t.width = t.height = e, t.dirty()) : (t = i(e), this._ssaoPass.setUniform("noiseTex", i(e))), this._ssaoPass.setUniform("noiseTexSize", [e, e])
        }, o.prototype.dispose = function(e) { this._targetTexture.dispose(e), this._ssaoTexture.dispose(e) }, e.exports = o
    }, function(e, t) { e.exports = "@export ecgl.ssr.main\n\n#define MAX_ITERATION 20;\n\nuniform sampler2D sourceTexture;\nuniform sampler2D gBufferTexture1;\nuniform sampler2D gBufferTexture2;\n\nuniform mat4 projection;\nuniform mat4 projectionInv;\nuniform mat4 viewInverseTranspose;\n\nuniform float maxRayDistance: 50;\n\nuniform float pixelStride: 16;\nuniform float pixelStrideZCutoff: 50; \nuniform float screenEdgeFadeStart: 0.9; \nuniform float eyeFadeStart : 0.2; uniform float eyeFadeEnd: 0.8; \nuniform float minGlossiness: 0.2; uniform float zThicknessThreshold: 10;\n\nuniform float nearZ;\nuniform vec2 viewportSize : VIEWPORT_SIZE;\n\nuniform float jitterOffset: 0;\n\nvarying vec2 v_Texcoord;\n\n#ifdef DEPTH_DECODE\n@import qtek.util.decode_float\n#endif\n\nfloat fetchDepth(sampler2D depthTexture, vec2 uv)\n{\n vec4 depthTexel = texture2D(depthTexture, uv);\n return depthTexel.r * 2.0 - 1.0;\n}\n\nfloat linearDepth(float depth)\n{\n if (projection[3][3] == 0.0) {\n return projection[3][2] / (depth * projection[2][3] - projection[2][2]);\n }\n else {\n return (depth - projection[3][2]) / projection[2][2];\n }\n}\n\nbool rayIntersectDepth(float rayZNear, float rayZFar, vec2 hitPixel)\n{\n if (rayZFar > rayZNear)\n {\n float t = rayZFar; rayZFar = rayZNear; rayZNear = t;\n }\n float cameraZ = linearDepth(fetchDepth(gBufferTexture2, hitPixel));\n return rayZFar <= cameraZ && rayZNear >= cameraZ - zThicknessThreshold;\n}\n\n\nbool traceScreenSpaceRay(\n vec3 rayOrigin, vec3 rayDir, float jitter,\n out vec2 hitPixel, out vec3 hitPoint, out float iterationCount\n)\n{\n float rayLength = ((rayOrigin.z + rayDir.z * maxRayDistance) > -nearZ)\n ? (-nearZ - rayOrigin.z) / rayDir.z : maxRayDistance;\n\n vec3 rayEnd = rayOrigin + rayDir * rayLength;\n\n vec4 H0 = projection * vec4(rayOrigin, 1.0);\n vec4 H1 = projection * vec4(rayEnd, 1.0);\n\n float k0 = 1.0 / H0.w, k1 = 1.0 / H1.w;\n\n vec3 Q0 = rayOrigin * k0, Q1 = rayEnd * k1;\n\n vec2 P0 = (H0.xy * k0 * 0.5 + 0.5) * viewportSize;\n vec2 P1 = (H1.xy * k1 * 0.5 + 0.5) * viewportSize;\n\n P1 += dot(P1 - P0, P1 - P0) < 0.0001 ? 0.01 : 0.0;\n vec2 delta = P1 - P0;\n\n bool permute = false;\n if (abs(delta.x) < abs(delta.y)) {\n permute = true;\n delta = delta.yx;\n P0 = P0.yx;\n P1 = P1.yx;\n }\n float stepDir = sign(delta.x);\n float invdx = stepDir / delta.x;\n\n vec3 dQ = (Q1 - Q0) * invdx;\n float dk = (k1 - k0) * invdx;\n\n vec2 dP = vec2(stepDir, delta.y * invdx);\n\n float strideScaler = 1.0 - min(1.0, -rayOrigin.z / pixelStrideZCutoff);\n float pixStride = 1.0 + strideScaler * pixelStride;\n\n dP *= pixStride; dQ *= pixStride; dk *= pixStride;\n\n vec4 pqk = vec4(P0, Q0.z, k0);\n vec4 dPQK = vec4(dP, dQ.z, dk);\n\n pqk += dPQK * jitter;\n float rayZFar = (dPQK.z * 0.5 + pqk.z) / (dPQK.w * 0.5 + pqk.w);\n float rayZNear;\n\n bool intersect = false;\n\n vec2 texelSize = 1.0 / viewportSize;\n\n iterationCount = 0.0;\n\n for (int i = 0; i < MAX_ITERATION; i++)\n {\n pqk += dPQK;\n\n rayZNear = rayZFar;\n rayZFar = (dPQK.z * 0.5 + pqk.z) / (dPQK.w * 0.5 + pqk.w);\n\n hitPixel = permute ? pqk.yx : pqk.xy;\n hitPixel *= texelSize;\n\n intersect = rayIntersectDepth(rayZNear, rayZFar, hitPixel);\n\n iterationCount += 1.0;\n\n if (intersect) {\n break;\n }\n }\n\n\n Q0.xy += dQ.xy * iterationCount;\n Q0.z = pqk.z;\n hitPoint = Q0 / pqk.w;\n\n return intersect;\n}\n\nfloat calculateAlpha(\n float iterationCount, float reflectivity,\n vec2 hitPixel, vec3 hitPoint, float dist, vec3 rayDir\n)\n{\n float alpha = clamp(reflectivity, 0.0, 1.0);\n alpha *= 1.0 - (iterationCount / float(MAX_ITERATION));\n vec2 hitPixelNDC = hitPixel * 2.0 - 1.0;\n float maxDimension = min(1.0, max(abs(hitPixelNDC.x), abs(hitPixelNDC.y)));\n alpha *= 1.0 - max(0.0, maxDimension - screenEdgeFadeStart) / (1.0 - screenEdgeFadeStart);\n\n float _eyeFadeStart = eyeFadeStart;\n float _eyeFadeEnd = eyeFadeEnd;\n if (_eyeFadeStart > _eyeFadeEnd) {\n float tmp = _eyeFadeEnd;\n _eyeFadeEnd = _eyeFadeStart;\n _eyeFadeStart = tmp;\n }\n\n float eyeDir = clamp(rayDir.z, _eyeFadeStart, _eyeFadeEnd);\n alpha *= 1.0 - (eyeDir - _eyeFadeStart) / (_eyeFadeEnd - _eyeFadeStart);\n\n alpha *= 1.0 - clamp(dist / maxRayDistance, 0.0, 1.0);\n\n return alpha;\n}\n\n@import qtek.util.rand\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec4 normalAndGloss = texture2D(gBufferTexture1, v_Texcoord);\n\n if (dot(normalAndGloss.rgb, vec3(1.0)) == 0.0) {\n discard;\n }\n\n float g = normalAndGloss.a;\n if (g <= minGlossiness) {\n discard;\n }\n\n float reflectivity = (g - minGlossiness) / (1.0 - minGlossiness);\n\n vec3 N = normalAndGloss.rgb * 2.0 - 1.0;\n N = normalize((viewInverseTranspose * vec4(N, 0.0)).xyz);\n\n vec4 projectedPos = vec4(v_Texcoord * 2.0 - 1.0, fetchDepth(gBufferTexture2, v_Texcoord), 1.0);\n vec4 pos = projectionInv * projectedPos;\n vec3 rayOrigin = pos.xyz / pos.w;\n\n vec3 rayDir = normalize(reflect(normalize(rayOrigin), N));\n vec2 hitPixel;\n vec3 hitPoint;\n float iterationCount;\n\n vec2 uv2 = v_Texcoord * viewportSize;\n float jitter = rand(fract(v_Texcoord + jitterOffset));\n\n bool intersect = traceScreenSpaceRay(rayOrigin, rayDir, jitter, hitPixel, hitPoint, iterationCount);\n\n float dist = distance(rayOrigin, hitPoint);\n\n float alpha = calculateAlpha(iterationCount, reflectivity, hitPixel, hitPoint, dist, rayDir) * float(intersect);\n\n vec3 hitNormal = texture2D(gBufferTexture1, hitPixel).rgb * 2.0 - 1.0;\n hitNormal = normalize((viewInverseTranspose * vec4(hitNormal, 0.0)).xyz);\n\n if (dot(hitNormal, rayDir) >= 0.0) {\n discard;\n }\n\n \n if (!intersect) {\n discard;\n }\n vec4 color = decodeHDR(texture2D(sourceTexture, hitPixel));\n gl_FragColor = encodeHDR(vec4(color.rgb * alpha, color.a));\n}\n@end\n\n@export ecgl.ssr.blur\n\nuniform sampler2D texture;\nuniform sampler2D gBufferTexture1;\n\nvarying vec2 v_Texcoord;\n\nuniform vec2 textureSize;\nuniform float blurSize : 4.0;\n\n#ifdef BLEND\nuniform sampler2D sourceTexture;\n#endif\n\n@import qtek.util.rgbm\n\n\nvoid main()\n{\n @import qtek.compositor.kernel.gaussian_13\n\n vec4 centerNTexel = texture2D(gBufferTexture1, v_Texcoord);\n float g = centerNTexel.a;\n float maxBlurSize = clamp(1.0 - g + 0.1, 0.0, 1.0) * blurSize;\n#ifdef VERTICAL\n vec2 off = vec2(0.0, maxBlurSize / textureSize.y);\n#else\n vec2 off = vec2(maxBlurSize / textureSize.x, 0.0);\n#endif\n\n vec2 coord = v_Texcoord;\n\n vec4 sum = vec4(0.0);\n float weightAll = 0.0;\n\n vec3 cN = centerNTexel.rgb * 2.0 - 1.0;\n for (int i = 0; i < 13; i++) {\n vec2 coord = clamp((float(i) - 6.0) * off + v_Texcoord, vec2(0.0), vec2(1.0));\n float w = gaussianKernel[i] * clamp(dot(cN, texture2D(gBufferTexture1, coord).rgb * 2.0 - 1.0), 0.0, 1.0);\n weightAll += w;\n sum += decodeHDR(texture2D(texture, coord)) * w;\n }\n\n#ifdef BLEND\n gl_FragColor = encodeHDR(\n sum / weightAll + decodeHDR(texture2D(sourceTexture, v_Texcoord))\n );\n#else\n gl_FragColor = encodeHDR(sum / weightAll);\n#endif\n}\n\n@end" }, function(e, t, r) {
        function n(e) { e = e || {}, this._ssrPass = new s({ fragment: u.source("ecgl.ssr.main"), clearColor: [0, 0, 0, 0] }), this._blurPass1 = new s({ fragment: u.source("ecgl.ssr.blur"), clearColor: [0, 0, 0, 0] }), this._blurPass2 = new s({ fragment: u.source("ecgl.ssr.blur"), clearColor: [0, 0, 0, 0] }), this._ssrPass.setUniform("gBufferTexture1", e.normalTexture), this._ssrPass.setUniform("gBufferTexture2", e.depthTexture), this._blurPass1.setUniform("gBufferTexture1", e.normalTexture), this._blurPass2.setUniform("gBufferTexture1", e.normalTexture), this._blurPass2.material.shader.define("fragment", "VERTICAL"), this._blurPass2.material.shader.define("fragment", "BLEND"), this._texture1 = new a({ type: o.HALF_FLOAT }), this._texture2 = new a({ type: o.HALF_FLOAT }), this._frameBuffer = new h }
        var i = r(9),
            a = (r(3), r(5)),
            o = r(6),
            s = r(12),
            u = r(7),
            h = r(10);
        r(39);
        u.import(r(162)), n.prototype.update = function(e, t, r, n) {
            var a = e.getWidth(),
                o = e.getHeight(),
                s = this._texture1,
                u = this._texture2;
            s.width = u.width = a, s.height = u.height = o;
            var h = this._frameBuffer,
                l = this._ssrPass,
                c = this._blurPass1,
                d = this._blurPass2,
                f = new i;
            i.transpose(f, t.worldTransform), l.setUniform("sourceTexture", r), l.setUniform("projection", t.projectionMatrix._array), l.setUniform("projectionInv", t.invProjectionMatrix._array), l.setUniform("viewInverseTranspose", f._array), l.setUniform("nearZ", t.near), l.setUniform("jitterOffset", n / 30);
            var p = [a, o];
            c.setUniform("textureSize", p), d.setUniform("textureSize", p), d.setUniform("sourceTexture", r), h.attach(u), h.bind(e), l.render(e), h.attach(s), c.setUniform("texture", u), c.render(e), h.attach(u), d.setUniform("texture", s), d.render(e), h.unbind(e)
        }, n.prototype.getTargetTexture = function() { return this._texture2 }, n.prototype.setParameter = function(e, t) { "maxIteration" === e ? this._ssrPass.material.shader.define("fragment", "MAX_ITERATION", t) : this._ssrPass.setUniform(e, t) }, n.prototype.dispose = function(e) { this._texture1.dispose(e), this._texture2.dispose(e), this._frameBuffer.dispose(e) }, e.exports = n
    }, function(e, t, r) {
        function n() {
            for (var e = [], t = 0; t < 30; t++) e.push([i(t, 2), i(t, 3)]);
            this._haltonSequence = e, this._frame = 0, this._sourceTex = new s, this._sourceFb = new o, this._sourceFb.attach(this._sourceTex), this._prevFrameTex = new s, this._outputTex = new s;
            var r = this._blendPass = new a({ fragment: u.source("qtek.compositor.blend") });
            r.material.shader.disableTexturesAll(), r.material.shader.enableTexture(["texture1", "texture2"]), this._blendFb = new o({ depthBuffer: !1 }), this._outputPass = new a({ fragment: u.source("qtek.compositor.output"), blendWithPrevious: !0 }), this._outputPass.material.shader.define("fragment", "OUTPUT_ALPHA"), this._outputPass.material.blend = function(e) { e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD), e.blendFuncSeparate(e.ONE, e.ONE_MINUS_SRC_ALPHA, e.ONE, e.ONE_MINUS_SRC_ALPHA) }
        }
        var i = r(39),
            a = r(12),
            o = r(10),
            s = r(5),
            u = r(7),
            h = r(9);
        r(3);
        n.prototype = {
            constructor: n,
            jitterProjection: function(e, t) {
                var r = e.viewport,
                    n = r.devicePixelRatio || e.getDevicePixelRatio(),
                    i = r.width * n,
                    a = r.height * n,
                    o = this._haltonSequence[this._frame],
                    s = new h;
                s._array[12] = (2 * o[0] - 1) / i, s._array[13] = (2 * o[1] - 1) / a, h.mul(t.projectionMatrix, s, t.projectionMatrix), h.invert(t.invProjectionMatrix, t.projectionMatrix)
            },
            resetFrame: function() { this._frame = 0 },
            getFrame: function() { return this._frame },
            getSourceFrameBuffer: function() { return this._sourceFb },
            resize: function(e, t) { this._sourceTex.width === e && this._sourceTex.height === t || (this._prevFrameTex.width = e, this._prevFrameTex.height = t, this._outputTex.width = e, this._outputTex.height = t, this._sourceTex.width = e, this._sourceTex.height = t, this._prevFrameTex.dirty(), this._outputTex.dirty(), this._sourceTex.dirty()) },
            isFinished: function() { return this._frame >= this._haltonSequence.length },
            render: function(e) {
                var t = this._blendPass;
                0 === this._frame ? (t.setUniform("weight1", 0), t.setUniform("weight2", 1)) : (t.setUniform("weight1", .9), t.setUniform("weight2", .1)), t.setUniform("texture1", this._prevFrameTex), t.setUniform("texture2", this._sourceTex), this._blendFb.attach(this._outputTex), this._blendFb.bind(e), t.render(e), this._blendFb.unbind(e), this._outputPass.setUniform("texture", this._outputTex), this._outputPass.render(e);
                var r = this._prevFrameTex;
                this._prevFrameTex = this._outputTex, this._outputTex = r, this._frame++
            },
            dispose: function(e) { this._sourceFb.dispose(e), this._blendFb.dispose(e), this._prevFrameTex.dispose(e), this._outputTex.dispose(e), this._sourceTex.dispose(e), this._outputPass.dispose(e), this._blendPass.dispose(e) }
        }, e.exports = n
    }, function(e, t) { e.exports = { type: "compositor", nodes: [{ name: "source", type: "texture", outputs: { color: {} } }, { name: "source_half", shader: "#source(qtek.compositor.downsample)", inputs: { texture: "source" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 2)", height: "expr(height * 1.0 / 2)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0, height * 1.0] )" } }, { name: "bright", shader: "#source(qtek.compositor.bright)", inputs: { texture: "source_half" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 2)", height: "expr(height * 1.0 / 2)", type: "HALF_FLOAT" } } }, parameters: { threshold: 2, scale: 4, textureSize: "expr([width * 1.0 / 2, height / 2])" } }, { name: "bright_downsample_4", shader: "#source(qtek.compositor.downsample)", inputs: { texture: "bright" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 4)", height: "expr(height * 1.0 / 4)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0 / 2, height / 2] )" } }, { name: "bright_downsample_8", shader: "#source(qtek.compositor.downsample)", inputs: { texture: "bright_downsample_4" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 8)", height: "expr(height * 1.0 / 8)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0 / 4, height / 4] )" } }, { name: "bright_downsample_16", shader: "#source(qtek.compositor.downsample)", inputs: { texture: "bright_downsample_8" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 16)", height: "expr(height * 1.0 / 16)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0 / 8, height / 8] )" } }, { name: "bright_downsample_32", shader: "#source(qtek.compositor.downsample)", inputs: { texture: "bright_downsample_16" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 32)", height: "expr(height * 1.0 / 32)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0 / 16, height / 16] )" } }, { name: "bright_upsample_16_blur_h", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_downsample_32" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 16)", height: "expr(height * 1.0 / 16)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 0, textureSize: "expr( [width * 1.0 / 32, height / 32] )" } }, { name: "bright_upsample_16_blur_v", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_upsample_16_blur_h" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 16)", height: "expr(height * 1.0 / 16)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 1, textureSize: "expr( [width * 1.0 / 32, height * 1.0 / 32] )" } }, { name: "bright_upsample_8_blur_h", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_downsample_16" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 8)", height: "expr(height * 1.0 / 8)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 0, textureSize: "expr( [width * 1.0 / 16, height * 1.0 / 16] )" } }, { name: "bright_upsample_8_blur_v", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_upsample_8_blur_h" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 8)", height: "expr(height * 1.0 / 8)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 1, textureSize: "expr( [width * 1.0 / 16, height * 1.0 / 16] )" } }, { name: "bright_upsample_8_blend", shader: "#source(qtek.compositor.blend)", inputs: { texture1: "bright_upsample_8_blur_v", texture2: "bright_upsample_16_blur_v" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 8)", height: "expr(height * 1.0 / 8)", type: "HALF_FLOAT" } } }, parameters: { weight1: .3, weight2: .7 } }, { name: "bright_upsample_4_blur_h", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_downsample_8" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 4)", height: "expr(height * 1.0 / 4)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 0, textureSize: "expr( [width * 1.0 / 8, height * 1.0 / 8] )" } }, { name: "bright_upsample_4_blur_v", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_upsample_4_blur_h" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 4)", height: "expr(height * 1.0 / 4)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 1, textureSize: "expr( [width * 1.0 / 8, height * 1.0 / 8] )" } }, { name: "bright_upsample_4_blend", shader: "#source(qtek.compositor.blend)", inputs: { texture1: "bright_upsample_4_blur_v", texture2: "bright_upsample_8_blend" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 4)", height: "expr(height * 1.0 / 4)", type: "HALF_FLOAT" } } }, parameters: { weight1: .3, weight2: .7 } }, { name: "bright_upsample_2_blur_h", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_downsample_4" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 2)", height: "expr(height * 1.0 / 2)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 0, textureSize: "expr( [width * 1.0 / 4, height * 1.0 / 4] )" } }, { name: "bright_upsample_2_blur_v", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_upsample_2_blur_h" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 2)", height: "expr(height * 1.0 / 2)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 1, textureSize: "expr( [width * 1.0 / 4, height * 1.0 / 4] )" } }, { name: "bright_upsample_2_blend", shader: "#source(qtek.compositor.blend)", inputs: { texture1: "bright_upsample_2_blur_v", texture2: "bright_upsample_4_blend" }, outputs: { color: { parameters: { width: "expr(width * 1.0 / 2)", height: "expr(height * 1.0 / 2)", type: "HALF_FLOAT" } } }, parameters: { weight1: .3, weight2: .7 } }, { name: "bright_upsample_full_blur_h", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 0, textureSize: "expr( [width * 1.0 / 2, height * 1.0 / 2] )" } }, { name: "bright_upsample_full_blur_v", shader: "#source(qtek.compositor.gaussian_blur)", inputs: { texture: "bright_upsample_full_blur_h" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } }, parameters: { blurSize: 1, blurDir: 1, textureSize: "expr( [width * 1.0 / 2, height * 1.0 / 2] )" } }, { name: "bloom_composite", shader: "#source(qtek.compositor.blend)", inputs: { texture1: "bright_upsample_full_blur_v", texture2: "bright_upsample_2_blend" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } }, parameters: { weight1: .3, weight2: .7 } }, { name: "coc", shader: "#source(ecgl.dof.coc)", outputs: { color: { parameters: { minFilter: "NEAREST", magFilter: "NEAREST", width: "expr(width * 1.0)", height: "expr(height * 1.0)" } } }, parameters: { focalDist: 50, focalRange: 30 } }, { name: "dof_far_blur", shader: "#source(ecgl.dof.diskBlur)", inputs: { texture: "source", coc: "coc" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0, height * 1.0] )" } }, { name: "dof_near_blur", shader: "#source(ecgl.dof.diskBlur)", inputs: { texture: "source", coc: "coc" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } }, parameters: { textureSize: "expr( [width * 1.0, height * 1.0] )" }, defines: { BLUR_NEARFIELD: null } }, { name: "dof_coc_blur", shader: "#source(ecgl.dof.diskBlur)", inputs: { texture: "coc" }, outputs: { color: { parameters: { minFilter: "NEAREST", magFilter: "NEAREST", width: "expr(width * 1.0)", height: "expr(height * 1.0)" } } }, parameters: { textureSize: "expr( [width * 1.0, height * 1.0] )" }, defines: { BLUR_COC: null } }, { name: "dof_composite", shader: "#source(ecgl.dof.composite)", inputs: { original: "source", blurred: "dof_far_blur", nearfield: "dof_near_blur", coc: "coc", nearcoc: "dof_coc_blur" }, outputs: { color: { parameters: { width: "expr(width * 1.0)", height: "expr(height * 1.0)", type: "HALF_FLOAT" } } } }, { name: "composite", shader: "#source(qtek.compositor.hdr.composite)", inputs: { texture: "source", bloom: "bloom_composite" }, defines: {} }, { name: "FXAA", shader: "#source(qtek.compositor.fxaa)", inputs: { texture: "composite" } }] } }, function(e, t) { e.exports = "@export ecgl.edge\n\nuniform sampler2D texture;\n\nuniform sampler2D normalTexture;\nuniform sampler2D depthTexture;\n\nuniform mat4 projectionInv;\n\nuniform vec2 textureSize;\n\nuniform vec4 edgeColor: [0,0,0,0.8];\n\nvarying vec2 v_Texcoord;\n\nvec3 packColor(vec2 coord) {\n float z = texture2D(depthTexture, coord).r * 2.0 - 1.0;\n vec4 p = vec4(v_Texcoord * 2.0 - 1.0, z, 1.0);\n vec4 p4 = projectionInv * p;\n\n return vec3(\n texture2D(normalTexture, coord).rg,\n -p4.z / p4.w / 5.0\n );\n}\n\nvoid main() {\n vec2 cc = v_Texcoord;\n vec3 center = packColor(cc);\n\n float size = clamp(1.0 - (center.z - 10.0) / 100.0, 0.0, 1.0) * 0.5;\n float dx = size / textureSize.x;\n float dy = size / textureSize.y;\n\n vec2 coord;\n vec3 topLeft = packColor(cc+vec2(-dx, -dy));\n vec3 top = packColor(cc+vec2(0.0, -dy));\n vec3 topRight = packColor(cc+vec2(dx, -dy));\n vec3 left = packColor(cc+vec2(-dx, 0.0));\n vec3 right = packColor(cc+vec2(dx, 0.0));\n vec3 bottomLeft = packColor(cc+vec2(-dx, dy));\n vec3 bottom = packColor(cc+vec2(0.0, dy));\n vec3 bottomRight = packColor(cc+vec2(dx, dy));\n\n vec3 v = -topLeft-2.0*top-topRight+bottomLeft+2.0*bottom+bottomRight;\n vec3 h = -bottomLeft-2.0*left-topLeft+bottomRight+2.0*right+topRight;\n\n float edge = sqrt(dot(h, h) + dot(v, v));\n\n edge = smoothstep(0.8, 1.0, edge);\n\n gl_FragColor = mix(texture2D(texture, v_Texcoord), vec4(edgeColor.rgb, 1.0), edgeColor.a * edge);\n}\n@end" }, function(e, t) { e.exports = [0, 0, -.321585265978, -.154972575841, .458126042375, .188473391593, .842080129861, .527766490688, .147304551086, -.659453822776, -.331943915203, -.940619700594, .0479226680259, .54812163202, .701581552186, -.709825561388, -.295436780218, .940589268233, -.901489676764, .237713156085, .973570876096, -.109899459384, -.866792314779, -.451805525005, .330975007087, .800048655954, -.344275183665, .381779221166, -.386139432542, -.437418421534, -.576478634965, -.0148463392551, .385798197415, -.262426961053, -.666302061145, .682427250835, -.628010632582, -.732836215494, .10163141741, -.987658134403, .711995289051, -.320024291314, .0296005138058, .950296523438, .0130612307608, -.351024443122, -.879596633704, -.10478487883, .435712737232, .504254490347, .779203817497, .206477676721, .388264289969, -.896736162545, -.153106280781, -.629203242522, -.245517550697, .657969239148, .126830499058, .26862328493, -.634888119007, -.302301223431, .617074219636, .779817204925] }, function(e, t, r) {
        function n(e, t) {
            if (e && e[t] && (e[t].normal || e[t].emphasis)) {
                var r = e[t].normal,
                    n = e[t].emphasis;
                r && (e[t] = r), n && (e.emphasis = e.emphasis || {}, e.emphasis[t] = n)
            }
        }

        function i(e) { n(e, "itemStyle"), n(e, "lineStyle"), n(e, "areaStyle"), n(e, "label") }

        function a(e) {
            e && (e instanceof Array || (e = [e]), o.util.each(e, function(e) {
                if (e.axisLabel) {
                    var t = e.axisLabel;
                    o.util.extend(t, t.textStyle), t.textStyle = null
                }
            }))
        }
        var o = r(0),
            s = ["bar3D", "line3D", "map3D", "scatter3D", "surface", "lines3D", "scatterGL", "scatter3D"];
        e.exports = function(e) { o.util.each(e.series, function(e) { o.util.indexOf(s, e.type) >= 0 && i(e) }), a(e.xAxis3D), a(e.yAxis3D), a(e.zAxis3D), a(e.grid3D), n(e.geo3D) }
    }, function(e, t, r) {
        function n(e) { return "_on" + e }
        var i = r(5),
            a = r(3),
            o = r(28),
            s = ["mousedown", "mouseup", "mousemove", "mouseover", "mouseout", "click", "dblclick", "contextmenu"],
            u = function(e) {
                var t = this;
                this._texture = new i({ anisotropic: 32, flipY: !1, surface: this, dispose: function(e) { t.dispose(), i.prototype.dispose.call(this, e) } }), s.forEach(function(e) { this[n(e)] = function(t) { t.triangle && this._meshes.forEach(function(r) { this.dispatchEvent(e, r, t.triangle, t.point) }, this) } }, this), this._meshes = [], e && this.setECharts(e), this.onupdate = null
            };
        u.prototype = {
            constructor: u,
            getTexture: function() { return this._texture },
            setECharts: function(e) {
                this._chart = e;
                var t = e.getDom();
                if (t instanceof HTMLCanvasElement) {
                    var r = this,
                        n = e.getZr(),
                        i = n.__oldRefreshImmediately || n.refreshImmediately;
                    n.refreshImmediately = function() { i.call(this), r._texture.dirty(), r.onupdate && r.onupdate() }, n.__oldRefreshImmediately = i
                } else console.error("ECharts must init on canvas if it is used as texture."), t = document.createElement("canvas");
                this._texture.image = t, this._texture.dirty(), this.onupdate && this.onupdate()
            },
            dispatchEvent: function() {
                var e = new a,
                    t = new a,
                    r = new a,
                    n = new o,
                    i = new o,
                    s = new o,
                    u = new o,
                    h = new a;
                return function(l, c, d, f) {
                    var p = c.geometry,
                        _ = p.attributes.position,
                        m = p.attributes.texcoord0,
                        g = a.dot,
                        v = a.cross;
                    _.get(d[0], e._array), _.get(d[1], t._array), _.get(d[2], r._array), m.get(d[0], n._array), m.get(d[1], i._array), m.get(d[2], s._array), v(h, t, r);
                    var y = g(e, h),
                        x = g(f, h) / y;
                    v(h, r, e);
                    var T = g(f, h) / y;
                    v(h, e, t);
                    var b = g(f, h) / y;
                    o.scale(u, n, x), o.scaleAndAdd(u, u, i, T), o.scaleAndAdd(u, u, s, b);
                    var w = u.x * this._chart.getWidth(),
                        E = u.y * this._chart.getHeight();
                    this._chart.getZr().handler.dispatch(l, { zrX: w, zrY: E })
                }
            }(),
            attachToMesh: function(e) { this._meshes.indexOf(e) >= 0 || (s.forEach(function(t) { e.on(t, this[n(t)], this) }, this), this._meshes.push(e)) },
            detachFromMesh: function(e) {
                var t = this._meshes.indexOf(e);
                t >= 0 && this._meshes.splice(t, 1), s.forEach(function(t) { e.off(t, this[n(t)]) }, this)
            },
            dispose: function() { this._meshes.forEach(function(e) { this.detachFromMesh(e) }, this) }
        }, e.exports = u
    }, function(e, t, r) {
        var n = r(8),
            i = (r(4), n.extend(function() { return { zr: null, viewGL: null, minZoom: .2, maxZoom: 5, _needsUpdate: !1, _dx: 0, _dy: 0, _zoom: 1 } }, function() { this._mouseDownHandler = this._mouseDownHandler.bind(this), this._mouseWheelHandler = this._mouseWheelHandler.bind(this), this._mouseMoveHandler = this._mouseMoveHandler.bind(this), this._mouseUpHandler = this._mouseUpHandler.bind(this), this._update = this._update.bind(this) }, {
                init: function() {
                    var e = this.zr;
                    e.on("mousedown", this._mouseDownHandler), e.on("mousewheel", this._mouseWheelHandler), e.on("globalout", this._mouseUpHandler), e.animation.on("frame", this._update)
                },
                setTarget: function(e) { this._target = e },
                setZoom: function(e) { this._zoom = Math.max(Math.min(e, this.maxZoom), this.minZoom), this._needsUpdate = !0 },
                setOffset: function(e) { this._dx = e[0], this._dy = e[1], this._needsUpdate = !0 },
                getZoom: function() { return this._zoom },
                getOffset: function() { return [this._dx, this._dy] },
                _update: function() {
                    if (this._target && this._needsUpdate) {
                        var e = this._target,
                            t = this._zoom;
                        e.position.x = this._dx, e.position.y = this._dy, e.scale.set(t, t, t), this.zr.refresh(), this._needsUpdate = !1, this.trigger("update")
                    }
                },
                _mouseDownHandler: function(e) {
                    if (!e.target) {
                        var t = e.offsetX,
                            r = e.offsetY;
                        if (!this.viewGL || this.viewGL.containPoint(t, r)) {
                            this.zr.on("mousemove", this._mouseMoveHandler), this.zr.on("mouseup", this._mouseUpHandler);
                            var n = this._convertPos(t, r);
                            this._x = n.x, this._y = n.y
                        }
                    }
                },
                _convertPos: function(e, t) {
                    var r = this.viewGL.camera,
                        n = this.viewGL.viewport;
                    return { x: (e - n.x) / n.width * (r.right - r.left) + r.left, y: (t - n.y) / n.height * (r.bottom - r.top) + r.top }
                },
                _mouseMoveHandler: function(e) {
                    var t = this._convertPos(e.offsetX, e.offsetY);
                    this._dx += t.x - this._x, this._dy += t.y - this._y, this._x = t.x, this._y = t.y, this._needsUpdate = !0
                },
                _mouseUpHandler: function(e) { this.zr.off("mousemove", this._mouseMoveHandler), this.zr.off("mouseup", this._mouseUpHandler) },
                _mouseWheelHandler: function(e) {
                    e = e.event;
                    var t = e.wheelDelta || -e.detail;
                    if (0 !== t) {
                        var r = e.offsetX,
                            n = e.offsetY;
                        if (!this.viewGL || this.viewGL.containPoint(r, n)) {
                            var i = t > 0 ? 1.1 : .9,
                                a = Math.max(Math.min(this._zoom * i, this.maxZoom), this.minZoom);
                            i = a / this._zoom;
                            var o = this._convertPos(r, n),
                                s = (o.x - this._dx) * (i - 1),
                                u = (o.y - this._dy) * (i - 1);
                            this._dx -= s, this._dy -= u, this._zoom = a, this._needsUpdate = !0
                        }
                    }
                },
                dispose: function() {
                    var e = this.zr;
                    e.off("mousedown", this._mouseDownHandler), e.off("mousemove", this._mouseMoveHandler), e.off("mouseup", this._mouseUpHandler), e.off("mousewheel", this._mouseWheelHandler), e.off("globalout", this._mouseUpHandler), e.animation.off("frame", this._update)
                }
            }));
        e.exports = i
    }, function(e, t, r) {
        var n = r(236),
            i = {
                _animators: null,
                getAnimators: function() { return this._animators = this._animators || [], this._animators },
                animate: function(e, t) {
                    this._animators = this._animators || [];
                    var r, i = this;
                    if (e) {
                        for (var a = e.split("."), o = i, s = 0, u = a.length; s < u; s++) o && (o = o[a[s]]);
                        o && (r = o)
                    } else r = i;
                    if (null == r) throw new Error("Target " + e + " not exists");
                    var h = this._animators,
                        l = new n(r, t),
                        c = this;
                    return l.during(function() { c.__zr && c.__zr.refresh() }).done(function() {
                        var e = h.indexOf(l);
                        e >= 0 && h.splice(e, 1)
                    }), h.push(l), this.__zr && this.__zr.animation.addAnimator(l), l
                },
                stopAnimation: function(e) { this._animators = this._animators || []; for (var t = this._animators, r = t.length, n = 0; n < r; n++) t[n].stop(e); return t.length = 0, this },
                addAnimatorsToZr: function(e) {
                    if (this._animators)
                        for (var t = 0; t < this._animators.length; t++) e.animation.addAnimator(this._animators[t])
                },
                removeAnimatorsFromZr: function(e) {
                    if (this._animators)
                        for (var t = 0; t < this._animators.length; t++) e.animation.removeAnimator(this._animators[t])
                }
            };
        e.exports = i
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r) {
            r = r || 2;
            var n = t && t.length,
                a = n ? t[0] * r : e.length,
                s = i(e, 0, a, r, !0),
                u = [];
            if (!s) return u;
            var h, l, d, f, p, _, m;
            if (n && (s = c(e, t, s, r)), e.length > 80 * r) {
                h = d = e[0], l = f = e[1];
                for (var g = r; g < a; g += r) p = e[g], _ = e[g + 1], p < h && (h = p), _ < l && (l = _), p > d && (d = p), _ > f && (f = _);
                m = Math.max(d - h, f - l)
            }
            return o(s, u, r, h, l, m), u
        }

        function i(e, t, r, n, i) {
            var a, o;
            if (i === D(e, t, r, n) > 0)
                for (a = t; a < r; a += n) o = N(a, e[a], e[a + 1], o);
            else
                for (a = r - n; a >= t; a -= n) o = N(a, e[a], e[a + 1], o);
            return o && b(o, o.next) && (C(o), o = o.next), o
        }

        function a(e, t) {
            if (!e) return e;
            t || (t = e);
            var r, n = e;
            do {
                if (r = !1, n.steiner || !b(n, n.next) && 0 !== T(n.prev, n, n.next)) n = n.next;
                else {
                    if (C(n), (n = t = n.prev) === n.next) return null;
                    r = !0
                }
            } while (r || n !== t);
            return t
        }

        function o(e, t, r, n, i, c, d) {
            if (e) {
                !d && c && _(e, n, i, c);
                for (var f, p, m = e; e.prev !== e.next;)
                    if (f = e.prev, p = e.next, c ? u(e, n, i, c) : s(e)) t.push(f.i / r), t.push(e.i / r), t.push(p.i / r), C(e), e = p.next, m = p.next;
                    else if ((e = p) === m) { d ? 1 === d ? (e = h(e, t, r), o(e, t, r, n, i, c, 2)) : 2 === d && l(e, t, r, n, i, c) : o(a(e), t, r, n, i, c, 1); break }
            }
        }

        function s(e) {
            var t = e.prev,
                r = e,
                n = e.next;
            if (T(t, r, n) >= 0) return !1;
            for (var i = e.next.next; i !== e.prev;) {
                if (y(t.x, t.y, r.x, r.y, n.x, n.y, i.x, i.y) && T(i.prev, i, i.next) >= 0) return !1;
                i = i.next
            }
            return !0
        }

        function u(e, t, r, n) {
            var i = e.prev,
                a = e,
                o = e.next;
            if (T(i, a, o) >= 0) return !1;
            for (var s = i.x < a.x ? i.x < o.x ? i.x : o.x : a.x < o.x ? a.x : o.x, u = i.y < a.y ? i.y < o.y ? i.y : o.y : a.y < o.y ? a.y : o.y, h = i.x > a.x ? i.x > o.x ? i.x : o.x : a.x > o.x ? a.x : o.x, l = i.y > a.y ? i.y > o.y ? i.y : o.y : a.y > o.y ? a.y : o.y, c = g(s, u, t, r, n), d = g(h, l, t, r, n), f = e.nextZ; f && f.z <= d;) {
                if (f !== e.prev && f !== e.next && y(i.x, i.y, a.x, a.y, o.x, o.y, f.x, f.y) && T(f.prev, f, f.next) >= 0) return !1;
                f = f.nextZ
            }
            for (f = e.prevZ; f && f.z >= c;) {
                if (f !== e.prev && f !== e.next && y(i.x, i.y, a.x, a.y, o.x, o.y, f.x, f.y) && T(f.prev, f, f.next) >= 0) return !1;
                f = f.prevZ
            }
            return !0
        }

        function h(e, t, r) {
            var n = e;
            do {
                var i = n.prev,
                    a = n.next.next;
                !b(i, a) && w(i, n, n.next, a) && S(i, a) && S(a, i) && (t.push(i.i / r), t.push(n.i / r), t.push(a.i / r), C(n), C(n.next), n = e = a), n = n.next
            } while (n !== e);
            return n
        }

        function l(e, t, r, n, i, s) {
            var u = e;
            do {
                for (var h = u.next.next; h !== u.prev;) {
                    if (u.i !== h.i && x(u, h)) { var l = M(u, h); return u = a(u, u.next), l = a(l, l.next), o(u, t, r, n, i, s), void o(l, t, r, n, i, s) }
                    h = h.next
                }
                u = u.next
            } while (u !== e)
        }

        function c(e, t, r, n) { var o, s, u, h, l, c = []; for (o = 0, s = t.length; o < s; o++) u = t[o] * n, h = o < s - 1 ? t[o + 1] * n : e.length, l = i(e, u, h, n, !1), l === l.next && (l.steiner = !0), c.push(v(l)); for (c.sort(d), o = 0; o < c.length; o++) f(c[o], r), r = a(r, r.next); return r }

        function d(e, t) { return e.x - t.x }

        function f(e, t) {
            if (t = p(e, t)) {
                var r = M(t, e);
                a(r, r.next)
            }
        }

        function p(e, t) {
            var r, n = t,
                i = e.x,
                a = e.y,
                o = -1 / 0;
            do {
                if (a <= n.y && a >= n.next.y && n.next.y !== n.y) {
                    var s = n.x + (a - n.y) * (n.next.x - n.x) / (n.next.y - n.y);
                    if (s <= i && s > o) {
                        if (o = s, s === i) { if (a === n.y) return n; if (a === n.next.y) return n.next }
                        r = n.x < n.next.x ? n : n.next
                    }
                }
                n = n.next
            } while (n !== t);
            if (!r) return null;
            if (i === o) return r.prev;
            var u, h = r,
                l = r.x,
                c = r.y,
                d = 1 / 0;
            for (n = r.next; n !== h;) i >= n.x && n.x >= l && i !== n.x && y(a < c ? i : o, a, l, c, a < c ? o : i, a, n.x, n.y) && ((u = Math.abs(a - n.y) / (i - n.x)) < d || u === d && n.x > r.x) && S(n, e) && (r = n, d = u), n = n.next;
            return r
        }

        function _(e, t, r, n) {
            var i = e;
            do { null === i.z && (i.z = g(i.x, i.y, t, r, n)), i.prevZ = i.prev, i.nextZ = i.next, i = i.next } while (i !== e);
            i.prevZ.nextZ = null, i.prevZ = null, m(i)
        }

        function m(e) {
            var t, r, n, i, a, o, s, u, h = 1;
            do {
                for (r = e, e = null, a = null, o = 0; r;) {
                    for (o++, n = r, s = 0, t = 0; t < h && (s++, n = n.nextZ); t++);
                    for (u = h; s > 0 || u > 0 && n;) 0 !== s && (0 === u || !n || r.z <= n.z) ? (i = r, r = r.nextZ, s--) : (i = n, n = n.nextZ, u--), a ? a.nextZ = i : e = i, i.prevZ = a, a = i;
                    r = n
                }
                a.nextZ = null, h *= 2
            } while (o > 1);
            return e
        }

        function g(e, t, r, n, i) { return e = 32767 * (e - r) / i, t = 32767 * (t - n) / i, e = 16711935 & (e | e << 8), e = 252645135 & (e | e << 4), e = 858993459 & (e | e << 2), e = 1431655765 & (e | e << 1), t = 16711935 & (t | t << 8), t = 252645135 & (t | t << 4), t = 858993459 & (t | t << 2), t = 1431655765 & (t | t << 1), e | t << 1 }

        function v(e) {
            var t = e,
                r = e;
            do { t.x < r.x && (r = t), t = t.next } while (t !== e);
            return r
        }

        function y(e, t, r, n, i, a, o, s) { return (i - o) * (t - s) - (e - o) * (a - s) >= 0 && (e - o) * (n - s) - (r - o) * (t - s) >= 0 && (r - o) * (a - s) - (i - o) * (n - s) >= 0 }

        function x(e, t) { return e.next.i !== t.i && e.prev.i !== t.i && !E(e, t) && S(e, t) && S(t, e) && A(e, t) }

        function T(e, t, r) { return (t.y - e.y) * (r.x - t.x) - (t.x - e.x) * (r.y - t.y) }

        function b(e, t) { return e.x === t.x && e.y === t.y }

        function w(e, t, r, n) { return !!(b(e, t) && b(r, n) || b(e, n) && b(r, t)) || T(e, t, r) > 0 != T(e, t, n) > 0 && T(r, n, e) > 0 != T(r, n, t) > 0 }

        function E(e, t) {
            var r = e;
            do {
                if (r.i !== e.i && r.next.i !== e.i && r.i !== t.i && r.next.i !== t.i && w(r, r.next, e, t)) return !0;
                r = r.next
            } while (r !== e);
            return !1
        }

        function S(e, t) { return T(e.prev, e, e.next) < 0 ? T(e, t, e.next) >= 0 && T(e, e.prev, t) >= 0 : T(e, t, e.prev) < 0 || T(e, e.next, t) < 0 }

        function A(e, t) {
            var r = e,
                n = !1,
                i = (e.x + t.x) / 2,
                a = (e.y + t.y) / 2;
            do { r.y > a != r.next.y > a && r.next.y !== r.y && i < (r.next.x - r.x) * (a - r.y) / (r.next.y - r.y) + r.x && (n = !n), r = r.next } while (r !== e);
            return n
        }

        function M(e, t) {
            var r = new L(e.i, e.x, e.y),
                n = new L(t.i, t.x, t.y),
                i = e.next,
                a = t.prev;
            return e.next = t, t.prev = e, r.next = i, i.prev = r, n.next = r, r.prev = n, a.next = n, n.prev = a, n
        }

        function N(e, t, r, n) { var i = new L(e, t, r); return n ? (i.next = n.next, i.prev = n, n.next.prev = i, n.next = i) : (i.prev = i, i.next = i), i }

        function C(e) { e.next.prev = e.prev, e.prev.next = e.next, e.prevZ && (e.prevZ.nextZ = e.nextZ), e.nextZ && (e.nextZ.prevZ = e.prevZ) }

        function L(e, t, r) { this.i = e, this.x = t, this.y = r, this.prev = null, this.next = null, this.z = null, this.prevZ = null, this.nextZ = null, this.steiner = !1 }

        function D(e, t, r, n) { for (var i = 0, a = t, o = r - n; a < r; a += n) i += (e[o] - e[a]) * (e[a + 1] + e[o + 1]), o = a; return i }
        e.exports = n, n.deviation = function(e, t, r, n) {
            var i = t && t.length,
                a = i ? t[0] * r : e.length,
                o = Math.abs(D(e, 0, a, r));
            if (i)
                for (var s = 0, u = t.length; s < u; s++) {
                    var h = t[s] * r,
                        l = s < u - 1 ? t[s + 1] * r : e.length;
                    o -= Math.abs(D(e, h, l, r))
                }
            var c = 0;
            for (s = 0; s < n.length; s += 3) {
                var d = n[s] * r,
                    f = n[s + 1] * r,
                    p = n[s + 2] * r;
                c += Math.abs((e[d] - e[p]) * (e[f + 1] - e[d + 1]) - (e[d] - e[f]) * (e[p + 1] - e[d + 1]))
            }
            return 0 === o && 0 === c ? 0 : Math.abs((c - o) / o)
        }
    }, function(e, t, r) {
        var n = r(0),
            i = r(34),
            a = r(50),
            o = r(13),
            s = r(1),
            u = s.vec3,
            h = s.mat3,
            l = o.extend(function() { return { attributes: { position: new o.Attribute("position", "float", 3, "POSITION"), normal: new o.Attribute("normal", "float", 3, "NORMAL"), color: new o.Attribute("color", "float", 4, "COLOR"), prevPosition: new o.Attribute("prevPosition", "float", 3), prevNormal: new o.Attribute("prevNormal", "float", 3) }, dynamic: !0, enableNormal: !1, bevelSize: 1, bevelSegments: 0, _dataIndices: null, _vertexOffset: 0, _triangleOffset: 0 } }, {
                resetOffset: function() { this._vertexOffset = 0, this._triangleOffset = 0 },
                setBarCount: function(e) {
                    var t = this.enableNormal,
                        r = this.getBarVertexCount() * e,
                        n = this.getBarTriangleCount() * e;
                    this.vertexCount !== r && (this.attributes.position.init(r), t ? this.attributes.normal.init(r) : this.attributes.normal.value = null, this.attributes.color.init(r)), this.triangleCount !== n && (this.indices = r > 65535 ? new Uint32Array(3 * n) : new Uint16Array(3 * n), this._dataIndices = new Uint32Array(r))
                },
                getBarVertexCount: function() { var e = this.bevelSize > 0 ? this.bevelSegments : 0; return e > 0 ? this._getBevelBarVertexCount(e) : this.enableNormal ? 24 : 8 },
                getBarTriangleCount: function() { var e = this.bevelSize > 0 ? this.bevelSegments : 0; return e > 0 ? this._getBevelBarTriangleCount(e) : 12 },
                _getBevelBarVertexCount: function(e) { return 4 * (e + 1) * (e + 1) * 2 },
                _getBevelBarTriangleCount: function(e) { return (4 * e + 3 + 1) * (2 * e + 1) * 2 + 4 },
                setColor: function(e, t) {
                    for (var r = this.getBarVertexCount(), n = r * e, i = r * (e + 1), a = n; a < i; a++) this.attributes.color.set(a, t);
                    this.dirtyAttribute("color")
                },
                getDataIndexOfVertex: function(e) { return this._dataIndices ? this._dataIndices[e] : null },
                addBar: function() {
                    for (var e = u.create, t = u.scaleAndAdd, r = e(), n = e(), i = e(), a = e(), o = e(), s = e(), h = e(), l = [], c = [], d = 0; d < 8; d++) l[d] = e();
                    for (var f = [
                            [0, 1, 5, 4],
                            [2, 3, 7, 6],
                            [4, 5, 6, 7],
                            [3, 2, 1, 0],
                            [0, 4, 7, 3],
                            [1, 2, 6, 5]
                        ], p = [0, 1, 2, 0, 2, 3], _ = [], d = 0; d < f.length; d++)
                        for (var m = f[d], g = 0; g < 2; g++) {
                            for (var v = [], y = 0; y < 3; y++) v.push(m[p[3 * g + y]]);
                            _.push(v)
                        }
                    return function(e, d, m, g, v, y) {
                        var x = this._vertexOffset;
                        if (this.bevelSize > 0 && this.bevelSegments > 0) this._addBevelBar(e, d, m, g, this.bevelSize, this.bevelSegments, v);
                        else {
                            u.copy(i, d), u.normalize(i, i), u.cross(a, m, i), u.normalize(a, a), u.cross(n, i, a), u.normalize(a, a), u.negate(o, n), u.negate(s, i), u.negate(h, a), t(l[0], e, n, g[0] / 2), t(l[0], l[0], a, g[2] / 2), t(l[1], e, n, g[0] / 2), t(l[1], l[1], h, g[2] / 2), t(l[2], e, o, g[0] / 2), t(l[2], l[2], h, g[2] / 2), t(l[3], e, o, g[0] / 2), t(l[3], l[3], a, g[2] / 2), t(r, e, i, g[1]), t(l[4], r, n, g[0] / 2), t(l[4], l[4], a, g[2] / 2), t(l[5], r, n, g[0] / 2), t(l[5], l[5], h, g[2] / 2), t(l[6], r, o, g[0] / 2), t(l[6], l[6], h, g[2] / 2), t(l[7], r, o, g[0] / 2), t(l[7], l[7], a, g[2] / 2);
                            var T = this.attributes;
                            if (this.enableNormal) {
                                c[0] = n, c[1] = o, c[2] = i, c[3] = s, c[4] = a, c[5] = h;
                                for (var b = this._vertexOffset, w = 0; w < f.length; w++) {
                                    for (var E = 3 * this._triangleOffset, S = 0; S < 6; S++) this.indices[E++] = b + p[S];
                                    b += 4, this._triangleOffset += 2
                                }
                                for (var w = 0; w < f.length; w++)
                                    for (var A = c[w], S = 0; S < 4; S++) {
                                        var M = f[w][S];
                                        T.position.set(this._vertexOffset, l[M]), T.normal.set(this._vertexOffset, A), T.color.set(this._vertexOffset++, v)
                                    }
                            } else {
                                for (var w = 0; w < _.length; w++) {
                                    for (var E = 3 * this._triangleOffset, S = 0; S < 3; S++) this.indices[E + S] = _[w][S] + this._vertexOffset;
                                    this._triangleOffset++
                                }
                                for (var w = 0; w < l.length; w++) T.position.set(this._vertexOffset, l[w]), T.color.set(this._vertexOffset++, v)
                            }
                        }
                        for (var N = this._vertexOffset, w = x; w < N; w++) this._dataIndices[w] = y
                    }
                }(),
                _addBevelBar: function() {
                    var e = u.create(),
                        t = u.create(),
                        r = u.create(),
                        n = h.create(),
                        i = [],
                        a = [1, -1, -1, 1],
                        o = [1, 1, -1, -1],
                        s = [2, 0];
                    return function(h, l, c, d, f, p, _) {
                        u.copy(t, l), u.normalize(t, t), u.cross(r, c, t), u.normalize(r, r), u.cross(e, t, r), u.normalize(r, r), n[0] = e[0], n[1] = e[1], n[2] = e[2], n[3] = t[0], n[4] = t[1], n[5] = t[2], n[6] = r[0], n[7] = r[1], n[8] = r[2], f = Math.min(d[0], d[2]) / 2 * f;
                        for (var m = 0; m < 3; m++) i[m] = Math.max(d[m] - 2 * f, 0);
                        for (var g = (d[0] - i[0]) / 2, v = (d[1] - i[1]) / 2, y = (d[2] - i[2]) / 2, x = [], T = [], b = this._vertexOffset, w = [], m = 0; m < 2; m++) {
                            w[m] = w[m] = [];
                            for (var E = 0; E <= p; E++)
                                for (var S = 0; S < 4; S++) {
                                    (0 === E && 0 === m || 1 === m && E === p) && w[m].push(b);
                                    for (var A = 0; A <= p; A++) {
                                        var M = A / p * Math.PI / 2 + Math.PI / 2 * S,
                                            N = E / p * Math.PI / 2 + Math.PI / 2 * m;
                                        T[0] = g * Math.cos(M) * Math.sin(N), T[1] = v * Math.cos(N), T[2] = y * Math.sin(M) * Math.sin(N), x[0] = T[0] + a[S] * i[0] / 2, x[1] = T[1] + v + s[m] * i[1] / 2, x[2] = T[2] + o[S] * i[2] / 2, Math.abs(g - v) < 1e-6 && Math.abs(v - y) < 1e-6 || (T[0] /= g * g, T[1] /= v * v, T[2] /= y * y), u.normalize(T, T), u.transformMat3(x, x, n), u.transformMat3(T, T, n), u.add(x, x, h), this.attributes.position.set(b, x), this.enableNormal && this.attributes.normal.set(b, T), this.attributes.color.set(b, _), b++
                                    }
                                }
                        }
                        for (var C = 4 * p + 3, L = 2 * p + 1, D = C + 1, S = 0; S < L; S++)
                            for (var m = 0; m <= C; m++) {
                                var I = S * D + m + this._vertexOffset,
                                    R = S * D + (m + 1) % D + this._vertexOffset,
                                    P = (S + 1) * D + (m + 1) % D + this._vertexOffset,
                                    O = (S + 1) * D + m + this._vertexOffset;
                                this.setTriangleIndices(this._triangleOffset++, [P, I, R]), this.setTriangleIndices(this._triangleOffset++, [P, O, I])
                            }
                        this.setTriangleIndices(this._triangleOffset++, [w[0][0], w[0][2], w[0][1]]), this.setTriangleIndices(this._triangleOffset++, [w[0][0], w[0][3], w[0][2]]), this.setTriangleIndices(this._triangleOffset++, [w[1][0], w[1][1], w[1][2]]), this.setTriangleIndices(this._triangleOffset++, [w[1][0], w[1][2], w[1][3]]), this._vertexOffset = b
                    }
                }()
            });
        n.util.defaults(l.prototype, i), n.util.defaults(l.prototype, a), e.exports = l
    }, function(e, t, r) {
        var n = r(13),
            i = r(1).vec2,
            a = r(0),
            o = r(34),
            s = [
                [0, 0],
                [1, 1]
            ],
            u = n.extend(function() { return { segmentScale: 4, dynamic: !0, useNativeLine: !0, attributes: { position: new n.Attribute("position", "float", 2, "POSITION"), normal: new n.Attribute("normal", "float", 2), offset: new n.Attribute("offset", "float", 1), color: new n.Attribute("color", "float", 4, "COLOR") } } }, {
                resetOffset: function() { this._vertexOffset = 0, this._faceOffset = 0, this._itemVertexOffsets = [] },
                setVertexCount: function(e) {
                    var t = this.attributes;
                    this.vertexCount !== e && (t.position.init(e), t.color.init(e), this.useNativeLine || (t.offset.init(e), t.normal.init(e)), e > 65535 ? this.indices instanceof Uint16Array && (this.indices = new Uint32Array(this.indices)) : this.indices instanceof Uint32Array && (this.indices = new Uint16Array(this.indices)))
                },
                setTriangleCount: function(e) { this.triangleCount !== e && (this.indices = 0 === e ? null : this.vertexCount > 65535 ? new Uint32Array(3 * e) : new Uint16Array(3 * e)) },
                _getCubicCurveApproxStep: function(e, t, r, n) { return 1 / (i.dist(e, t) + i.dist(r, t) + i.dist(n, r) + 1) * this.segmentScale },
                getCubicCurveVertexCount: function(e, t, r, n) {
                    var i = this._getCubicCurveApproxStep(e, t, r, n),
                        a = Math.ceil(1 / i);
                    return this.useNativeLine ? 2 * a : 2 * a + 2
                },
                getCubicCurveTriangleCount: function(e, t, r, n) {
                    var i = this._getCubicCurveApproxStep(e, t, r, n),
                        a = Math.ceil(1 / i);
                    return this.useNativeLine ? 0 : 2 * a
                },
                getLineVertexCount: function() { return this.getPolylineVertexCount(s) },
                getLineTriangleCount: function() { return this.getPolylineTriangleCount(s) },
                getPolylineVertexCount: function(e) {
                    var t = "number" != typeof e[0],
                        r = t ? e.length : e.length / 2;
                    return this.useNativeLine ? 2 * (r - 1) : 2 * (r - 1) + 2
                },
                getPolylineTriangleCount: function(e) {
                    var t = "number" != typeof e[0],
                        r = t ? e.length : e.length / 2;
                    return this.useNativeLine ? 0 : 2 * (r - 1)
                },
                addCubicCurve: function(e, t, r, n, i, a) {
                    null == a && (a = 1);
                    for (var o = e[0], s = e[1], u = t[0], h = t[1], l = r[0], c = r[1], d = n[0], f = n[1], p = this._getCubicCurveApproxStep(e, t, r, n), _ = p * p, m = _ * p, g = 3 * p, v = 3 * _, y = 6 * _, x = 6 * m, T = o - 2 * u + l, b = s - 2 * h + c, w = 3 * (u - l) - o + d, E = 3 * (h - c) - s + f, S = o, A = s, M = (u - o) * g + T * v + w * m, N = (h - s) * g + b * v + E * m, C = T * y + w * x, L = b * y + E * x, D = w * x, I = E * x, R = 0, P = 0, O = Math.ceil(1 / p), F = new Float32Array(3 * (O + 1)), F = [], B = 0, P = 0; P < O + 1; P++) F[B++] = S, F[B++] = A, S += M, A += N, M += C, N += L, C += D, L += I, (R += p) > 1 && (S = M > 0 ? Math.min(S, d) : Math.max(S, d), A = N > 0 ? Math.min(A, f) : Math.max(A, f));
                    this.addPolyline(F, i, a, !1)
                },
                addLine: function(e, t, r, n) { this.addPolyline([e, t], r, n, !1) },
                addPolyline: function(e, t, r, n) {
                    if (e.length) {
                        this._itemVertexOffsets.push(this._vertexOffset);
                        var a = "number" != typeof e[0],
                            o = this.attributes.position,
                            s = this.attributes.color,
                            u = this.attributes.offset,
                            h = this.attributes.normal,
                            l = this.indices;
                        null == r && (r = 1);
                        for (var c, d = this._vertexOffset, f = a ? e.length : e.length / 2, p = f, _ = [], m = [], g = [], v = i.create(), y = i.create(), x = i.create(), T = i.create(), b = 0; b < p; b++) {
                            if (a) _ = e[b], c = n ? t[b] : t;
                            else {
                                var w = 2 * b;
                                if (_ = _ || [], _[0] = e[w], _[1] = e[w + 1], n) {
                                    var E = 4 * b;
                                    c = c || [], c[0] = t[E], c[1] = t[E + 1], c[2] = t[E + 2], c[3] = t[E + 3]
                                } else c = t
                            }
                            if (this.useNativeLine) b > 1 && (o.copy(d, d - 1), s.copy(d, d - 1), d++);
                            else {
                                var S;
                                if (b < p - 1) {
                                    if (a) i.copy(m, e[b + 1]);
                                    else {
                                        var w = 2 * (b + 1);
                                        m = m || [], m[0] = e[w], m[1] = e[w + 1]
                                    }
                                    if (b > 0) {
                                        i.sub(v, _, g), i.sub(y, m, _), i.normalize(v, v), i.normalize(y, y), i.add(T, v, y), i.normalize(T, T);
                                        var A = r / 2 * Math.min(1 / i.dot(v, T), 2);
                                        x[0] = -T[1], x[1] = T[0], S = A
                                    } else i.sub(v, m, _), i.normalize(v, v), x[0] = -v[1], x[1] = v[0], S = r / 2
                                } else i.sub(v, _, g), i.normalize(v, v), x[0] = -v[1], x[1] = v[0], S = r / 2;
                                h.set(d, x), h.set(d + 1, x), u.set(d, S), u.set(d + 1, -S), i.copy(g, _), o.set(d, _), o.set(d + 1, _), s.set(d, c), s.set(d + 1, c), d += 2
                            }
                            if (this.useNativeLine) s.set(d, c), o.set(d, _), d++;
                            else if (b > 0) {
                                var M = 3 * this._faceOffset,
                                    l = this.indices;
                                l[M] = d - 4, l[M + 1] = d - 3, l[M + 2] = d - 2, l[M + 3] = d - 3, l[M + 4] = d - 1, l[M + 5] = d - 2, this._faceOffset += 2
                            }
                        }
                        this._vertexOffset = d
                    }
                },
                setItemColor: function(e, t) {
                    for (var r = this._itemVertexOffsets[e], n = e < this._itemVertexOffsets.length - 1 ? this._itemVertexOffsets[e + 1] : this._vertexOffset, i = r; i < n; i++) this.attributes.color.set(i, t);
                    this.dirty("color")
                }
            });
        a.util.defaults(u.prototype, o), e.exports = u
    }, function(e, t, r) {
        var n = r(13),
            i = r(1).vec3,
            a = r(0),
            o = r(34),
            s = n.extend(function() { return { segmentScale: 1, useNativeLine: !0, attributes: { position: new n.Attribute("position", "float", 3, "POSITION"), normal: new n.Attribute("normal", "float", 3, "NORMAL"), color: new n.Attribute("color", "float", 4, "COLOR") } } }, {
                resetOffset: function() { this._vertexOffset = 0, this._faceOffset = 0 },
                setQuadCount: function(e) {
                    var t = this.attributes,
                        r = this.getQuadVertexCount() * e,
                        n = this.getQuadTriangleCount() * e;
                    this.vertexCount !== r && (t.position.init(r), t.normal.init(r), t.color.init(r)), this.triangleCount !== n && (this.indices = r > 65535 ? new Uint32Array(3 * n) : new Uint16Array(3 * n))
                },
                getQuadVertexCount: function() { return 4 },
                getQuadTriangleCount: function() { return 2 },
                addQuad: function() {
                    var e = i.create(),
                        t = i.create(),
                        r = i.create(),
                        n = [0, 3, 1, 3, 2, 1];
                    return function(a, o) {
                        var s = this.attributes.position,
                            u = this.attributes.normal,
                            h = this.attributes.color;
                        i.sub(e, a[1], a[0]), i.sub(t, a[2], a[1]), i.cross(r, e, t), i.normalize(r, r);
                        for (var l = 0; l < 4; l++) s.set(this._vertexOffset + l, a[l]), h.set(this._vertexOffset + l, o), u.set(this._vertexOffset + l, r);
                        for (var c = 3 * this._faceOffset, l = 0; l < 6; l++) this.indices[c + l] = n[l] + this._vertexOffset;
                        this._vertexOffset += 4, this._faceOffset += 2
                    }
                }()
            });
        a.util.defaults(s.prototype, o), e.exports = s
    }, function(e, t, r) {
        var n = r(0),
            i = r(13),
            a = r(34),
            o = [0, 1, 2, 0, 2, 3],
            s = i.extend(function() { return { attributes: { position: new i.Attribute("position", "float", 3, "POSITION"), texcoord: new i.Attribute("texcoord", "float", 2, "TEXCOORD_0"), offset: new i.Attribute("offset", "float", 2), color: new i.Attribute("color", "float", 4, "COLOR") } } }, {
                resetOffset: function() { this._vertexOffset = 0, this._faceOffset = 0 },
                setSpriteCount: function(e) {
                    this._spriteCount = e;
                    var t = 4 * e,
                        r = 2 * e;
                    this.vertexCount !== t && (this.attributes.position.init(t), this.attributes.offset.init(t), this.attributes.color.init(t)), this.triangleCount !== r && (this.indices = t > 65535 ? new Uint32Array(3 * r) : new Uint16Array(3 * r))
                },
                setSpriteAlign: function(e, t, r, n, i) {
                    null == r && (r = "left"), null == n && (n = "top");
                    var a, o, s, u;
                    switch (i = i || 0, r) {
                        case "left":
                            a = i, s = t[0] + i;
                            break;
                        case "center":
                        case "middle":
                            a = -t[0] / 2, s = t[0] / 2;
                            break;
                        case "right":
                            a = -t[0] - i, s = -i
                    }
                    switch (n) {
                        case "bottom":
                            o = i, u = t[1] + i;
                            break;
                        case "middle":
                            o = -t[1] / 2, u = t[1] / 2;
                            break;
                        case "top":
                            o = -t[1] - i, u = -i
                    }
                    var h = 4 * e,
                        l = this.attributes.offset;
                    l.set(h, [a, u]), l.set(h + 1, [s, u]), l.set(h + 2, [s, o]), l.set(h + 3, [a, o])
                },
                addSprite: function(e, t, r, n, i, a) {
                    var s = this._vertexOffset;
                    this.setSprite(this._vertexOffset / 4, e, t, r, n, i, a);
                    for (var u = 0; u < o.length; u++) this.indices[3 * this._faceOffset + u] = o[u] + s;
                    return this._faceOffset += 2, this._vertexOffset += 4, s / 4
                },
                setSprite: function(e, t, r, n, i, a, o) {
                    for (var s = 4 * e, u = this.attributes, h = 0; h < 4; h++) u.position.set(s + h, t);
                    var l = u.texcoord;
                    l.set(s, [n[0][0], n[0][1]]), l.set(s + 1, [n[1][0], n[0][1]]), l.set(s + 2, [n[1][0], n[1][1]]), l.set(s + 3, [n[0][0], n[1][1]]), this.setSpriteAlign(e, r, i, a, o)
                }
            });
        n.util.defaults(s.prototype, a), e.exports = s
    }, function(e, t, r) {
        var n = r(1).vec3,
            i = r(66);
        e.exports = {
            needsSortVertices: function() { return this.sortVertices },
            needsSortVerticesProgressively: function() { return this.needsSortVertices() && this.vertexCount >= 2e4 },
            doSortVertices: function(e, t) {
                var r = this.indices,
                    i = n.create();
                if (!r) { r = this.indices = this.vertexCount > 65535 ? new Uint32Array(this.vertexCount) : new Uint16Array(this.vertexCount); for (var a = 0; a < r.length; a++) r[a] = a }
                if (0 === t) {
                    var o = this.attributes.position,
                        e = e._array,
                        s = 0;
                    this._zList && this._zList.length === this.vertexCount || (this._zList = new Float32Array(this.vertexCount));
                    for (var u, a = 0; a < this.vertexCount; a++) {
                        o.get(a, i);
                        var h = n.sqrDist(i, e);
                        isNaN(h) && (h = 1e7, s++), 0 === a ? (u = h, h = 0) : h -= u, this._zList[a] = h
                    }
                    this._noneCount = s
                }
                if (this.vertexCount < 2e4) 0 === t && this._simpleSort(this._noneCount / this.vertexCount > .05);
                else
                    for (var a = 0; a < 3; a++) this._progressiveQuickSort(3 * t + a);
                this.dirtyIndices()
            },
            _simpleSort: function(e) {
                function t(e, t) { return r[t] - r[e] }
                var r = this._zList,
                    n = this.indices;
                e ? Array.prototype.sort.call(n, t) : i.sort(n, t, 0, n.length - 1)
            },
            _progressiveQuickSort: function(e) {
                var t = this._zList,
                    r = this.indices;
                this._quickSort = this._quickSort || new i, this._quickSort.step(r, function(e, r) { return t[r] - t[e] }, e)
            }
        }
    }, function(e, t) { e.exports = "@export ecgl.color.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\n@import ecgl.common.uv.header\n\nattribute vec2 texcoord : TEXCOORD_0;\nattribute vec3 position: POSITION;\n\n@import ecgl.common.wireframe.vertexHeader\n\n#ifdef VERTEX_COLOR\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n#endif\n\n#ifdef VERTEX_ANIMATION\nattribute vec3 prevPosition;\nuniform float percent : 1.0;\n#endif\n\nvoid main()\n{\n#ifdef VERTEX_ANIMATION\n vec3 pos = mix(prevPosition, position, percent);\n#else\n vec3 pos = position;\n#endif\n\n gl_Position = worldViewProjection * vec4(pos, 1.0);\n\n @import ecgl.common.uv.main\n\n#ifdef VERTEX_COLOR\n v_Color = a_Color;\n#endif\n\n @import ecgl.common.wireframe.vertexMain\n\n}\n\n@end\n\n@export ecgl.color.fragment\n\n#define LAYER_DIFFUSEMAP_COUNT 0\n#define LAYER_EMISSIVEMAP_COUNT 0\n\nuniform sampler2D diffuseMap;\nuniform sampler2D detailMap;\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\n\n@import ecgl.common.layers.header\n\n@import ecgl.common.uv.fragmentHeader\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.util.srgb\n\nvoid main()\n{\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(color);\n#else\n gl_FragColor = color;\n#endif\n\n#ifdef VERTEX_COLOR\n gl_FragColor *= v_Color;\n#endif\n\n @import ecgl.common.albedo.main\n\n @import ecgl.common.diffuseLayer.main\n\n gl_FragColor *= albedoTexel;\n\n @import ecgl.common.emissiveLayer.main\n\n @import ecgl.common.wireframe.fragmentMain\n\n}\n@end" }, function(e, t) { e.exports = "\n@export ecgl.common.transformUniforms\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform mat4 worldInverseTranspose : WORLDINVERSETRANSPOSE;\nuniform mat4 world : WORLD;\n@end\n\n@export ecgl.common.attributes\nattribute vec3 position : POSITION;\nattribute vec2 texcoord : TEXCOORD_0;\nattribute vec3 normal : NORMAL;\n@end\n\n@export ecgl.common.uv.header\nuniform vec2 uvRepeat : [1.0, 1.0];\nuniform vec2 uvOffset : [0.0, 0.0];\nuniform vec2 detailUvRepeat : [1.0, 1.0];\nuniform vec2 detailUvOffset : [0.0, 0.0];\n\nvarying vec2 v_Texcoord;\nvarying vec2 v_DetailTexcoord;\n@end\n\n@export ecgl.common.uv.main\nv_Texcoord = texcoord * uvRepeat + uvOffset;\nv_DetailTexcoord = texcoord * detailUvRepeat + detailUvOffset;\n@end\n\n@export ecgl.common.uv.fragmentHeader\nvarying vec2 v_Texcoord;\nvarying vec2 v_DetailTexcoord;\n@end\n\n\n@export ecgl.common.albedo.main\n\n vec4 albedoTexel = vec4(1.0);\n#ifdef DIFFUSEMAP_ENABLED\n albedoTexel = texture2D(diffuseMap, v_Texcoord);\n #ifdef SRGB_DECODE\n albedoTexel = sRGBToLinear(albedoTexel);\n #endif\n#endif\n\n#ifdef DETAILMAP_ENABLED\n vec4 detailTexel = texture2D(detailMap, v_DetailTexcoord);\n #ifdef SRGB_DECODE\n detailTexel = sRGBToLinear(detailTexel);\n #endif\n albedoTexel.rgb = mix(albedoTexel.rgb, detailTexel.rgb, detailTexel.a);\n albedoTexel.a = detailTexel.a + (1.0 - detailTexel.a) * albedoTexel.a;\n#endif\n\n@end\n\n@export ecgl.common.wireframe.vertexHeader\n\n#ifdef WIREFRAME_QUAD\nattribute vec4 barycentric;\nvarying vec4 v_Barycentric;\n#elif defined(WIREFRAME_TRIANGLE)\nattribute vec3 barycentric;\nvarying vec3 v_Barycentric;\n#endif\n\n@end\n\n@export ecgl.common.wireframe.vertexMain\n\n#if defined(WIREFRAME_QUAD) || defined(WIREFRAME_TRIANGLE)\n v_Barycentric = barycentric;\n#endif\n\n@end\n\n\n@export ecgl.common.wireframe.fragmentHeader\n\nuniform float wireframeLineWidth : 1;\nuniform vec4 wireframeLineColor: [0, 0, 0, 0.5];\n\n#ifdef WIREFRAME_QUAD\nvarying vec4 v_Barycentric;\nfloat edgeFactor () {\n vec4 d = fwidth(v_Barycentric);\n vec4 a4 = smoothstep(vec4(0.0), d * wireframeLineWidth, v_Barycentric);\n return min(min(min(a4.x, a4.y), a4.z), a4.w);\n}\n#elif defined(WIREFRAME_TRIANGLE)\nvarying vec3 v_Barycentric;\nfloat edgeFactor () {\n vec3 d = fwidth(v_Barycentric);\n vec3 a3 = smoothstep(vec3(0.0), d * wireframeLineWidth, v_Barycentric);\n return min(min(a3.x, a3.y), a3.z);\n}\n#endif\n\n@end\n\n\n@export ecgl.common.wireframe.fragmentMain\n\n#if defined(WIREFRAME_QUAD) || defined(WIREFRAME_TRIANGLE)\n if (wireframeLineWidth > 0.) {\n vec4 lineColor = wireframeLineColor;\n#ifdef SRGB_DECODE\n lineColor = sRGBToLinear(lineColor);\n#endif\n\n gl_FragColor.rgb = mix(gl_FragColor.rgb, lineColor.rgb, (1.0 - edgeFactor()) * lineColor.a);\n }\n#endif\n@end\n\n\n\n\n@export ecgl.common.bumpMap.header\n\n#ifdef BUMPMAP_ENABLED\nuniform sampler2D bumpMap;\nuniform float bumpScale : 1.0;\n\n\nvec3 bumpNormal(vec3 surfPos, vec3 surfNormal, vec3 baseNormal)\n{\n vec2 dSTdx = dFdx(v_Texcoord);\n vec2 dSTdy = dFdy(v_Texcoord);\n\n float Hll = bumpScale * texture2D(bumpMap, v_Texcoord).x;\n float dHx = bumpScale * texture2D(bumpMap, v_Texcoord + dSTdx).x - Hll;\n float dHy = bumpScale * texture2D(bumpMap, v_Texcoord + dSTdy).x - Hll;\n\n vec3 vSigmaX = dFdx(surfPos);\n vec3 vSigmaY = dFdy(surfPos);\n vec3 vN = surfNormal;\n\n vec3 R1 = cross(vSigmaY, vN);\n vec3 R2 = cross(vN, vSigmaX);\n\n float fDet = dot(vSigmaX, R1);\n\n vec3 vGrad = sign(fDet) * (dHx * R1 + dHy * R2);\n return normalize(abs(fDet) * baseNormal - vGrad);\n\n}\n#endif\n\n@end\n\n@export ecgl.common.normalMap.vertexHeader\n\n#ifdef NORMALMAP_ENABLED\nattribute vec4 tangent : TANGENT;\nvarying vec3 v_Tangent;\nvarying vec3 v_Bitangent;\n#endif\n\n@end\n\n@export ecgl.common.normalMap.vertexMain\n\n#ifdef NORMALMAP_ENABLED\n if (dot(tangent, tangent) > 0.0) {\n v_Tangent = normalize((worldInverseTranspose * vec4(tangent.xyz, 0.0)).xyz);\n v_Bitangent = normalize(cross(v_Normal, v_Tangent) * tangent.w);\n }\n#endif\n\n@end\n\n\n@export ecgl.common.normalMap.fragmentHeader\n\n#ifdef NORMALMAP_ENABLED\nuniform sampler2D normalMap;\nvarying vec3 v_Tangent;\nvarying vec3 v_Bitangent;\n#endif\n\n@end\n\n@export ecgl.common.normalMap.fragmentMain\n#ifdef NORMALMAP_ENABLED\n if (dot(v_Tangent, v_Tangent) > 0.0) {\n vec3 normalTexel = texture2D(normalMap, v_DetailTexcoord).xyz;\n if (dot(normalTexel, normalTexel) > 0.0) { N = normalTexel * 2.0 - 1.0;\n mat3 tbn = mat3(v_Tangent, v_Bitangent, v_Normal);\n N = normalize(tbn * N);\n }\n }\n#endif\n@end\n\n\n\n@export ecgl.common.vertexAnimation.header\n\n#ifdef VERTEX_ANIMATION\nattribute vec3 prevPosition;\nattribute vec3 prevNormal;\nuniform float percent;\n#endif\n\n@end\n\n@export ecgl.common.vertexAnimation.main\n\n#ifdef VERTEX_ANIMATION\n vec3 pos = mix(prevPosition, position, percent);\n vec3 norm = mix(prevNormal, normal, percent);\n#else\n vec3 pos = position;\n vec3 norm = normal;\n#endif\n\n@end\n\n\n@export ecgl.common.ssaoMap.header\n#ifdef SSAOMAP_ENABLED\nuniform sampler2D ssaoMap;\nuniform vec4 viewport : VIEWPORT;\n#endif\n@end\n\n@export ecgl.common.ssaoMap.main\n float ao = 1.0;\n#ifdef SSAOMAP_ENABLED\n ao = texture2D(ssaoMap, (gl_FragCoord.xy - viewport.xy) / viewport.zw).r;\n#endif\n@end\n\n\n\n\n@export ecgl.common.diffuseLayer.header\n\n#if (LAYER_DIFFUSEMAP_COUNT > 0)\nuniform float layerDiffuseIntensity[LAYER_DIFFUSEMAP_COUNT];\nuniform sampler2D layerDiffuseMap[LAYER_DIFFUSEMAP_COUNT];\n#endif\n\n@end\n\n@export ecgl.common.emissiveLayer.header\n\n#if (LAYER_EMISSIVEMAP_COUNT > 0)\nuniform float layerEmissionIntensity[LAYER_EMISSIVEMAP_COUNT];\nuniform sampler2D layerEmissiveMap[LAYER_EMISSIVEMAP_COUNT];\n#endif\n\n@end\n\n@export ecgl.common.layers.header\n@import ecgl.common.diffuseLayer.header\n@import ecgl.common.emissiveLayer.header\n@end\n\n@export ecgl.common.diffuseLayer.main\n\n#if (LAYER_DIFFUSEMAP_COUNT > 0)\n for (int _idx_ = 0; _idx_ < LAYER_DIFFUSEMAP_COUNT; _idx_++) {{\n float intensity = layerDiffuseIntensity[_idx_];\n vec4 texel2 = texture2D(layerDiffuseMap[_idx_], v_Texcoord);\n #ifdef SRGB_DECODE\n texel2 = sRGBToLinear(texel2);\n #endif\n albedoTexel.rgb = mix(albedoTexel.rgb, texel2.rgb * intensity, texel2.a);\n albedoTexel.a = texel2.a + (1.0 - texel2.a) * albedoTexel.a;\n }}\n#endif\n\n@end\n\n@export ecgl.common.emissiveLayer.main\n\n#if (LAYER_EMISSIVEMAP_COUNT > 0)\n for (int _idx_ = 0; _idx_ < LAYER_EMISSIVEMAP_COUNT; _idx_++)\n {{\n vec4 texel2 = texture2D(layerEmissiveMap[_idx_], v_Texcoord) * layerEmissionIntensity[_idx_];\n #ifdef SRGB_DECODE\n texel2 = sRGBToLinear(texel2);\n #endif\n float intensity = layerEmissionIntensity[_idx_];\n gl_FragColor.rgb += texel2.rgb * texel2.a * intensity;\n }}\n#endif\n\n@end\n" }, function(e, t) { e.exports = "\n@export ecgl.displayShadow.vertex\n\n@import ecgl.common.transformUniforms\n\n@import ecgl.common.uv.header\n\n@import ecgl.common.attributes\n\nvarying vec3 v_WorldPosition;\n\nvarying vec3 v_Normal;\n\nvoid main()\n{\n @import ecgl.common.uv.main\n v_Normal = normalize((worldInverseTranspose * vec4(normal, 0.0)).xyz);\n \n v_WorldPosition = (world * vec4(position, 1.0)).xyz;\n gl_Position = worldViewProjection * vec4(position, 1.0);\n}\n\n@end\n\n\n@export ecgl.displayShadow.fragment\n\n@import ecgl.common.uv.fragmentHeader\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nuniform float roughness: 0.2;\n\n#ifdef DIRECTIONAL_LIGHT_COUNT\n@import qtek.header.directional_light\n#endif\n\n@import ecgl.common.ssaoMap.header\n\n@import qtek.plugin.compute_shadow_map\n\nvoid main()\n{\n float shadow = 1.0;\n\n @import ecgl.common.ssaoMap.main\n\n#if defined(DIRECTIONAL_LIGHT_COUNT) && defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n float shadowContribsDir[DIRECTIONAL_LIGHT_COUNT];\n if(shadowEnabled)\n {\n computeShadowOfDirectionalLights(v_WorldPosition, shadowContribsDir);\n }\n for (int i = 0; i < DIRECTIONAL_LIGHT_COUNT; i++) {\n shadow = min(shadow, shadowContribsDir[i] * 0.5 + 0.5);\n }\n#endif\n\n shadow *= 0.5 + ao * 0.5;\n shadow = clamp(shadow, 0.0, 1.0);\n\n gl_FragColor = vec4(vec3(0.0), 1.0 - shadow);\n}\n\n@end" }, function(e, t) { e.exports = "@export ecgl.hatching.vertex\n\n@import ecgl.realistic.vertex\n\n@end\n\n\n@export ecgl.hatching.fragment\n\n#define NORMAL_UP_AXIS 1\n#define NORMAL_FRONT_AXIS 2\n\n@import ecgl.common.uv.fragmentHeader\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nuniform vec4 color : [0.0, 0.0, 0.0, 1.0];\nuniform vec4 paperColor : [1.0, 1.0, 1.0, 1.0];\n\nuniform mat4 viewInverse : VIEWINVERSE;\n\n#ifdef AMBIENT_LIGHT_COUNT\n@import qtek.header.ambient_light\n#endif\n#ifdef AMBIENT_SH_LIGHT_COUNT\n@import qtek.header.ambient_sh_light\n#endif\n\n#ifdef DIRECTIONAL_LIGHT_COUNT\n@import qtek.header.directional_light\n#endif\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\n\n\n@import ecgl.common.ssaoMap.header\n\n@import ecgl.common.bumpMap.header\n\n@import qtek.util.srgb\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.plugin.compute_shadow_map\n\nuniform sampler2D hatch1;\nuniform sampler2D hatch2;\nuniform sampler2D hatch3;\nuniform sampler2D hatch4;\nuniform sampler2D hatch5;\nuniform sampler2D hatch6;\n\nfloat shade(in float tone) {\n vec4 c = vec4(1. ,1., 1., 1.);\n float step = 1. / 6.;\n vec2 uv = v_DetailTexcoord;\n if (tone <= step / 2.0) {\n c = mix(vec4(0.), texture2D(hatch6, uv), 12. * tone);\n }\n else if (tone <= step) {\n c = mix(texture2D(hatch6, uv), texture2D(hatch5, uv), 6. * tone);\n }\n if(tone > step && tone <= 2. * step){\n c = mix(texture2D(hatch5, uv), texture2D(hatch4, uv) , 6. * (tone - step));\n }\n if(tone > 2. * step && tone <= 3. * step){\n c = mix(texture2D(hatch4, uv), texture2D(hatch3, uv), 6. * (tone - 2. * step));\n }\n if(tone > 3. * step && tone <= 4. * step){\n c = mix(texture2D(hatch3, uv), texture2D(hatch2, uv), 6. * (tone - 3. * step));\n }\n if(tone > 4. * step && tone <= 5. * step){\n c = mix(texture2D(hatch2, uv), texture2D(hatch1, uv), 6. * (tone - 4. * step));\n }\n if(tone > 5. * step){\n c = mix(texture2D(hatch1, uv), vec4(1.), 6. * (tone - 5. * step));\n }\n\n return c.r;\n}\n\nconst vec3 w = vec3(0.2125, 0.7154, 0.0721);\n\nvoid main()\n{\n#ifdef SRGB_DECODE\n vec4 inkColor = sRGBToLinear(color);\n#else\n vec4 inkColor = color;\n#endif\n\n#ifdef VERTEX_COLOR\n #ifdef SRGB_DECODE\n inkColor *= sRGBToLinear(v_Color);\n #else\n inkColor *= v_Color;\n #endif\n#endif\n\n vec3 N = v_Normal;\n#ifdef DOUBLE_SIDED\n vec3 eyePos = viewInverse[3].xyz;\n vec3 V = normalize(eyePos - v_WorldPosition);\n\n if (dot(N, V) < 0.0) {\n N = -N;\n }\n#endif\n\n float tone = 0.0;\n\n float ambientFactor = 1.0;\n\n#ifdef BUMPMAP_ENABLED\n N = bumpNormal(v_WorldPosition, v_Normal, N);\n ambientFactor = dot(v_Normal, N);\n#endif\n\n vec3 N2 = vec3(N.x, N[NORMAL_UP_AXIS], N[NORMAL_FRONT_AXIS]);\n \n @import ecgl.common.ssaoMap.main\n\n#ifdef AMBIENT_LIGHT_COUNT\n for(int i = 0; i < AMBIENT_LIGHT_COUNT; i++)\n {\n tone += dot(ambientLightColor[i], w) * ambientFactor * ao;\n }\n#endif\n#ifdef AMBIENT_SH_LIGHT_COUNT\n for(int _idx_ = 0; _idx_ < AMBIENT_SH_LIGHT_COUNT; _idx_++)\n {{\n tone += dot(calcAmbientSHLight(_idx_, N2) * ambientSHLightColor[_idx_], w) * ao;\n }}\n#endif\n#ifdef DIRECTIONAL_LIGHT_COUNT\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n float shadowContribsDir[DIRECTIONAL_LIGHT_COUNT];\n if(shadowEnabled)\n {\n computeShadowOfDirectionalLights(v_WorldPosition, shadowContribsDir);\n }\n#endif\n for(int i = 0; i < DIRECTIONAL_LIGHT_COUNT; i++)\n {\n vec3 lightDirection = -directionalLightDirection[i];\n float lightTone = dot(directionalLightColor[i], w);\n\n float shadowContrib = 1.0;\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n if (shadowEnabled)\n {\n shadowContrib = shadowContribsDir[i];\n }\n#endif\n\n float ndl = dot(N, normalize(lightDirection)) * shadowContrib;\n\n tone += lightTone * clamp(ndl, 0.0, 1.0);\n }\n#endif\n\n gl_FragColor = mix(inkColor, paperColor, shade(clamp(tone, 0.0, 1.0)));\n }\n@end\n" }, function(e, t) { e.exports = "@export ecgl.labels.vertex\n\nattribute vec3 position: POSITION;\nattribute vec2 texcoord: TEXCOORD_0;\nattribute vec2 offset;\n#ifdef VERTEX_COLOR\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n#endif\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform vec4 viewport : VIEWPORT;\n\nvarying vec2 v_Texcoord;\n\nvoid main()\n{\n vec4 proj = worldViewProjection * vec4(position, 1.0);\n\n vec2 screen = (proj.xy / abs(proj.w) + 1.0) * 0.5 * viewport.zw;\n\n screen += offset;\n\n proj.xy = (screen / viewport.zw - 0.5) * 2.0 * abs(proj.w);\n gl_Position = proj;\n#ifdef VERTEX_COLOR\n v_Color = a_Color;\n#endif\n v_Texcoord = texcoord;\n}\n@end\n\n\n@export ecgl.labels.fragment\n\nuniform vec3 color : [1.0, 1.0, 1.0];\nuniform float alpha : 1.0;\nuniform sampler2D textureAtlas;\nuniform vec2 uvScale: [1.0, 1.0];\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\nvarying float v_Miter;\n\nvarying vec2 v_Texcoord;\n\nvoid main()\n{\n gl_FragColor = vec4(color, alpha) * texture2D(textureAtlas, v_Texcoord * uvScale);\n#ifdef VERTEX_COLOR\n gl_FragColor *= v_Color;\n#endif\n}\n\n@end" }, function(e, t) { e.exports = "/**\n * http: */\n\n@export ecgl.lambert.vertex\n\n@import ecgl.common.transformUniforms\n\n@import ecgl.common.uv.header\n\n\n@import ecgl.common.attributes\n\n@import ecgl.common.wireframe.vertexHeader\n\n#ifdef VERTEX_COLOR\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n#endif\n\n\n@import ecgl.common.vertexAnimation.header\n\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nvoid main()\n{\n @import ecgl.common.uv.main\n\n @import ecgl.common.vertexAnimation.main\n\n\n gl_Position = worldViewProjection * vec4(pos, 1.0);\n\n v_Normal = normalize((worldInverseTranspose * vec4(norm, 0.0)).xyz);\n v_WorldPosition = (world * vec4(pos, 1.0)).xyz;\n\n#ifdef VERTEX_COLOR\n v_Color = a_Color;\n#endif\n\n @import ecgl.common.wireframe.vertexMain\n}\n\n@end\n\n\n@export ecgl.lambert.fragment\n\n#define LAYER_DIFFUSEMAP_COUNT 0\n#define LAYER_EMISSIVEMAP_COUNT 0\n\n#define NORMAL_UP_AXIS 1\n#define NORMAL_FRONT_AXIS 2\n\n@import ecgl.common.uv.fragmentHeader\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nuniform sampler2D diffuseMap;\nuniform sampler2D detailMap;\n\n@import ecgl.common.layers.header\n\nuniform float emissionIntensity: 1.0;\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nuniform mat4 viewInverse : VIEWINVERSE;\n\n#ifdef AMBIENT_LIGHT_COUNT\n@import qtek.header.ambient_light\n#endif\n#ifdef AMBIENT_SH_LIGHT_COUNT\n@import qtek.header.ambient_sh_light\n#endif\n\n#ifdef DIRECTIONAL_LIGHT_COUNT\n@import qtek.header.directional_light\n#endif\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\n\n\n@import ecgl.common.ssaoMap.header\n\n@import ecgl.common.bumpMap.header\n\n@import qtek.util.srgb\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.plugin.compute_shadow_map\n\nvoid main()\n{\n#ifdef SRGB_DECODE\n gl_FragColor = sRGBToLinear(color);\n#else\n gl_FragColor = color;\n#endif\n\n#ifdef VERTEX_COLOR\n #ifdef SRGB_DECODE\n gl_FragColor *= sRGBToLinear(v_Color);\n #else\n gl_FragColor *= v_Color;\n #endif\n#endif\n\n @import ecgl.common.albedo.main\n\n @import ecgl.common.diffuseLayer.main\n\n gl_FragColor *= albedoTexel;\n\n vec3 N = v_Normal;\n#ifdef DOUBLE_SIDED\n vec3 eyePos = viewInverse[3].xyz;\n vec3 V = normalize(eyePos - v_WorldPosition);\n\n if (dot(N, V) < 0.0) {\n N = -N;\n }\n#endif\n\n float ambientFactor = 1.0;\n\n#ifdef BUMPMAP_ENABLED\n N = bumpNormal(v_WorldPosition, v_Normal, N);\n ambientFactor = dot(v_Normal, N);\n#endif\n\n vec3 N2 = vec3(N.x, N[NORMAL_UP_AXIS], N[NORMAL_FRONT_AXIS]);\n\n vec3 diffuseColor = vec3(0.0, 0.0, 0.0);\n\n @import ecgl.common.ssaoMap.main\n\n#ifdef AMBIENT_LIGHT_COUNT\n for(int i = 0; i < AMBIENT_LIGHT_COUNT; i++)\n {\n diffuseColor += ambientLightColor[i] * ambientFactor * ao;\n }\n#endif\n#ifdef AMBIENT_SH_LIGHT_COUNT\n for(int _idx_ = 0; _idx_ < AMBIENT_SH_LIGHT_COUNT; _idx_++)\n {{\n diffuseColor += calcAmbientSHLight(_idx_, N2) * ambientSHLightColor[_idx_] * ao;\n }}\n#endif\n#ifdef DIRECTIONAL_LIGHT_COUNT\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n float shadowContribsDir[DIRECTIONAL_LIGHT_COUNT];\n if(shadowEnabled)\n {\n computeShadowOfDirectionalLights(v_WorldPosition, shadowContribsDir);\n }\n#endif\n for(int i = 0; i < DIRECTIONAL_LIGHT_COUNT; i++)\n {\n vec3 lightDirection = -directionalLightDirection[i];\n vec3 lightColor = directionalLightColor[i];\n\n float shadowContrib = 1.0;\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n if (shadowEnabled)\n {\n shadowContrib = shadowContribsDir[i];\n }\n#endif\n\n float ndl = dot(N, normalize(lightDirection)) * shadowContrib;\n\n diffuseColor += lightColor * clamp(ndl, 0.0, 1.0);\n }\n#endif\n\n gl_FragColor.rgb *= diffuseColor;\n\n @import ecgl.common.emissiveLayer.main\n\n @import ecgl.common.wireframe.fragmentMain\n}\n\n@end" }, function(e, t) { e.exports = "@export ecgl.lines2D.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec2 position: POSITION;\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n\n#ifdef POSITIONTEXTURE_ENABLED\nuniform sampler2D positionTexture;\n#endif\n\nvoid main()\n{\n gl_Position = worldViewProjection * vec4(position, -10.0, 1.0);\n\n v_Color = a_Color;\n}\n\n@end\n\n@export ecgl.lines2D.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nvarying vec4 v_Color;\n\nvoid main()\n{\n gl_FragColor = color * v_Color;\n}\n@end\n\n\n@export ecgl.meshLines2D.vertex\n\nattribute vec2 position: POSITION;\nattribute vec2 normal;\nattribute float offset;\nattribute vec4 a_Color : COLOR;\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform vec4 viewport : VIEWPORT;\n\nvarying vec4 v_Color;\nvarying float v_Miter;\n\nvoid main()\n{\n vec4 p2 = worldViewProjection * vec4(position + normal, -10.0, 1.0);\n gl_Position = worldViewProjection * vec4(position, -10.0, 1.0);\n\n p2.xy /= p2.w;\n gl_Position.xy /= gl_Position.w;\n\n vec2 N = normalize(p2.xy - gl_Position.xy);\n gl_Position.xy += N * offset / viewport.zw * 2.0;\n\n gl_Position.xy *= gl_Position.w;\n\n v_Color = a_Color;\n}\n@end\n\n\n@export ecgl.meshLines2D.fragment\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nvarying vec4 v_Color;\nvarying float v_Miter;\n\nvoid main()\n{\n gl_FragColor = color * v_Color;\n}\n\n@end" }, function(e, t) { e.exports = "@export ecgl.normal.vertex\n\n@import ecgl.common.transformUniforms\n\n@import ecgl.common.uv.header\n\n@import ecgl.common.attributes\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\n@import ecgl.common.normalMap.vertexHeader\n\n@import ecgl.common.vertexAnimation.header\n\nvoid main()\n{\n\n @import ecgl.common.vertexAnimation.main\n\n @import ecgl.common.uv.main\n\n v_Normal = normalize((worldInverseTranspose * vec4(normal, 0.0)).xyz);\n v_WorldPosition = (world * vec4(pos, 1.0)).xyz;\n\n @import ecgl.common.normalMap.vertexMain\n\n gl_Position = worldViewProjection * vec4(pos, 1.0);\n\n}\n\n\n@end\n\n\n@export ecgl.normal.fragment\n\n#define ROUGHNESS_CHANEL 0\n\nuniform bool useBumpMap;\nuniform bool useRoughnessMap;\nuniform bool doubleSide;\nuniform float roughness;\n\n@import ecgl.common.uv.fragmentHeader\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nuniform mat4 viewInverse : VIEWINVERSE;\n\n@import ecgl.common.normalMap.fragmentHeader\n@import ecgl.common.bumpMap.header\n\nuniform sampler2D roughnessMap;\n\nvoid main()\n{\n vec3 N = v_Normal;\n \n bool flipNormal = false;\n if (doubleSide) {\n vec3 eyePos = viewInverse[3].xyz;\n vec3 V = normalize(eyePos - v_WorldPosition);\n\n if (dot(N, V) < 0.0) {\n flipNormal = true;\n }\n }\n\n @import ecgl.common.normalMap.fragmentMain\n\n if (useBumpMap) {\n N = bumpNormal(v_WorldPosition, v_Normal, N);\n }\n\n float g = 1.0 - roughness;\n\n if (useRoughnessMap) {\n float g2 = 1.0 - texture2D(roughnessMap, v_DetailTexcoord)[ROUGHNESS_CHANEL];\n g = clamp(g2 + (g - 0.5) * 2.0, 0.0, 1.0);\n }\n\n if (flipNormal) {\n N = -N;\n }\n\n gl_FragColor.rgb = (N.xyz + 1.0) * 0.5;\n gl_FragColor.a = g;\n}\n@end" }, function(e, t) { e.exports = "@export ecgl.realistic.vertex\n\n@import ecgl.common.transformUniforms\n\n@import ecgl.common.uv.header\n\n@import ecgl.common.attributes\n\n\n@import ecgl.common.wireframe.vertexHeader\n\n#ifdef VERTEX_COLOR\nattribute vec4 a_Color : COLOR;\nvarying vec4 v_Color;\n#endif\n\n#ifdef NORMALMAP_ENABLED\nattribute vec4 tangent : TANGENT;\nvarying vec3 v_Tangent;\nvarying vec3 v_Bitangent;\n#endif\n\n@import ecgl.common.vertexAnimation.header\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nvoid main()\n{\n\n @import ecgl.common.uv.main\n\n @import ecgl.common.vertexAnimation.main\n\n gl_Position = worldViewProjection * vec4(pos, 1.0);\n\n v_Normal = normalize((worldInverseTranspose * vec4(norm, 0.0)).xyz);\n v_WorldPosition = (world * vec4(pos, 1.0)).xyz;\n\n#ifdef VERTEX_COLOR\n v_Color = a_Color;\n#endif\n\n#ifdef NORMALMAP_ENABLED\n v_Tangent = normalize((worldInverseTranspose * vec4(tangent.xyz, 0.0)).xyz);\n v_Bitangent = normalize(cross(v_Normal, v_Tangent) * tangent.w);\n#endif\n\n @import ecgl.common.wireframe.vertexMain\n\n}\n\n@end\n\n\n\n@export ecgl.realistic.fragment\n\n#define LAYER_DIFFUSEMAP_COUNT 0\n#define LAYER_EMISSIVEMAP_COUNT 0\n#define PI 3.14159265358979\n#define ROUGHNESS_CHANEL 0\n#define METALNESS_CHANEL 1\n\n#define NORMAL_UP_AXIS 1\n#define NORMAL_FRONT_AXIS 2\n\n#ifdef VERTEX_COLOR\nvarying vec4 v_Color;\n#endif\n\n@import ecgl.common.uv.fragmentHeader\n\nvarying vec3 v_Normal;\nvarying vec3 v_WorldPosition;\n\nuniform sampler2D diffuseMap;\n\nuniform sampler2D detailMap;\nuniform sampler2D metalnessMap;\nuniform sampler2D roughnessMap;\n\n@import ecgl.common.layers.header\n\nuniform float emissionIntensity: 1.0;\n\nuniform vec4 color : [1.0, 1.0, 1.0, 1.0];\n\nuniform float metalness : 0.0;\nuniform float roughness : 0.5;\n\nuniform mat4 viewInverse : VIEWINVERSE;\n\n#ifdef AMBIENT_LIGHT_COUNT\n@import qtek.header.ambient_light\n#endif\n\n#ifdef AMBIENT_SH_LIGHT_COUNT\n@import qtek.header.ambient_sh_light\n#endif\n\n#ifdef AMBIENT_CUBEMAP_LIGHT_COUNT\n@import qtek.header.ambient_cubemap_light\n#endif\n\n#ifdef DIRECTIONAL_LIGHT_COUNT\n@import qtek.header.directional_light\n#endif\n\n@import ecgl.common.normalMap.fragmentHeader\n\n@import ecgl.common.ssaoMap.header\n\n@import ecgl.common.bumpMap.header\n\n@import qtek.util.srgb\n\n@import qtek.util.rgbm\n\n@import ecgl.common.wireframe.fragmentHeader\n\n@import qtek.plugin.compute_shadow_map\n\nvec3 F_Schlick(float ndv, vec3 spec) {\n return spec + (1.0 - spec) * pow(1.0 - ndv, 5.0);\n}\n\nfloat D_Phong(float g, float ndh) {\n float a = pow(8192.0, g);\n return (a + 2.0) / 8.0 * pow(ndh, a);\n}\n\nvoid main()\n{\n vec4 albedoColor = color;\n\n vec3 eyePos = viewInverse[3].xyz;\n vec3 V = normalize(eyePos - v_WorldPosition);\n#ifdef VERTEX_COLOR\n #ifdef SRGB_DECODE\n albedoColor *= sRGBToLinear(v_Color);\n #else\n albedoColor *= v_Color;\n #endif\n#endif\n\n @import ecgl.common.albedo.main\n\n @import ecgl.common.diffuseLayer.main\n\n albedoColor *= albedoTexel;\n\n float m = metalness;\n\n#ifdef METALNESSMAP_ENABLED\n float m2 = texture2D(metalnessMap, v_DetailTexcoord)[METALNESS_CHANEL];\n m = clamp(m2 + (m - 0.5) * 2.0, 0.0, 1.0);\n#endif\n\n vec3 baseColor = albedoColor.rgb;\n albedoColor.rgb = baseColor * (1.0 - m);\n vec3 specFactor = mix(vec3(0.04), baseColor, m);\n\n float g = 1.0 - roughness;\n\n#ifdef ROUGHNESSMAP_ENABLED\n float g2 = 1.0 - texture2D(roughnessMap, v_DetailTexcoord)[ROUGHNESS_CHANEL];\n g = clamp(g2 + (g - 0.5) * 2.0, 0.0, 1.0);\n#endif\n\n vec3 N = v_Normal;\n\n#ifdef DOUBLE_SIDED\n if (dot(N, V) < 0.0) {\n N = -N;\n }\n#endif\n\n float ambientFactor = 1.0;\n\n#ifdef BUMPMAP_ENABLED\n N = bumpNormal(v_WorldPosition, v_Normal, N);\n ambientFactor = dot(v_Normal, N);\n#endif\n\n@import ecgl.common.normalMap.fragmentMain\n\n vec3 N2 = vec3(N.x, N[NORMAL_UP_AXIS], N[NORMAL_FRONT_AXIS]);\n\n vec3 diffuseTerm = vec3(0.0);\n vec3 specularTerm = vec3(0.0);\n\n float ndv = clamp(dot(N, V), 0.0, 1.0);\n vec3 fresnelTerm = F_Schlick(ndv, specFactor);\n\n @import ecgl.common.ssaoMap.main\n\n#ifdef AMBIENT_LIGHT_COUNT\n for(int _idx_ = 0; _idx_ < AMBIENT_LIGHT_COUNT; _idx_++)\n {{\n diffuseTerm += ambientLightColor[_idx_] * ambientFactor * ao;\n }}\n#endif\n\n#ifdef AMBIENT_SH_LIGHT_COUNT\n for(int _idx_ = 0; _idx_ < AMBIENT_SH_LIGHT_COUNT; _idx_++)\n {{\n diffuseTerm += calcAmbientSHLight(_idx_, N2) * ambientSHLightColor[_idx_] * ao;\n }}\n#endif\n\n#ifdef DIRECTIONAL_LIGHT_COUNT\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n float shadowContribsDir[DIRECTIONAL_LIGHT_COUNT];\n if(shadowEnabled)\n {\n computeShadowOfDirectionalLights(v_WorldPosition, shadowContribsDir);\n }\n#endif\n for(int _idx_ = 0; _idx_ < DIRECTIONAL_LIGHT_COUNT; _idx_++)\n {{\n vec3 L = -directionalLightDirection[_idx_];\n vec3 lc = directionalLightColor[_idx_];\n\n vec3 H = normalize(L + V);\n float ndl = clamp(dot(N, normalize(L)), 0.0, 1.0);\n float ndh = clamp(dot(N, H), 0.0, 1.0);\n\n float shadowContrib = 1.0;\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n if (shadowEnabled)\n {\n shadowContrib = shadowContribsDir[_idx_];\n }\n#endif\n\n vec3 li = lc * ndl * shadowContrib;\n\n diffuseTerm += li;\n specularTerm += li * fresnelTerm * D_Phong(g, ndh);\n }}\n#endif\n\n\n#ifdef AMBIENT_CUBEMAP_LIGHT_COUNT\n vec3 L = reflect(-V, N);\n L = vec3(L.x, L[NORMAL_UP_AXIS], L[NORMAL_FRONT_AXIS]);\n float rough2 = clamp(1.0 - g, 0.0, 1.0);\n float bias2 = rough2 * 5.0;\n vec2 brdfParam2 = texture2D(ambientCubemapLightBRDFLookup[0], vec2(rough2, ndv)).xy;\n vec3 envWeight2 = specFactor * brdfParam2.x + brdfParam2.y;\n vec3 envTexel2;\n for(int _idx_ = 0; _idx_ < AMBIENT_CUBEMAP_LIGHT_COUNT; _idx_++)\n {{\n envTexel2 = RGBMDecode(textureCubeLodEXT(ambientCubemapLightCubemap[_idx_], L, bias2), 51.5);\n specularTerm += ambientCubemapLightColor[_idx_] * envTexel2 * envWeight2 * ao;\n }}\n#endif\n\n gl_FragColor.rgb = albedoColor.rgb * diffuseTerm + specularTerm;\n gl_FragColor.a = albedoColor.a;\n\n#ifdef SRGB_ENCODE\n gl_FragColor = linearTosRGB(gl_FragColor);\n#endif\n\n @import ecgl.common.emissiveLayer.main\n\n @import ecgl.common.wireframe.fragmentMain\n}\n\n@end" }, function(e, t) { e.exports = "@export ecgl.sm.depth.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position : POSITION;\n\n#ifdef VERTEX_ANIMATION\nattribute vec3 prevPosition;\nuniform float percent : 1.0;\n#endif\n\nvarying vec4 v_ViewPosition;\n\nvoid main(){\n\n#ifdef VERTEX_ANIMATION\n vec3 pos = mix(prevPosition, position, percent);\n#else\n vec3 pos = position;\n#endif\n\n v_ViewPosition = worldViewProjection * vec4(pos, 1.0);\n gl_Position = v_ViewPosition;\n\n}\n@end\n\n\n\n@export ecgl.sm.depth.fragment\n\n@import qtek.sm.depth.fragment\n\n@end" }, function(e, t, r) {
        function n(e, t, r) {
            var t = t || document.createElement("canvas");
            t.width = e, t.height = e;
            var n = t.getContext("2d");
            return r && r(n), t
        }

        function i(e, t, r, n) {
            o.util.isArray(t) || (t = [t, t]);
            var i = s.getMarginByStyle(r, n),
                a = t[0] + i.left + i.right,
                u = t[1] + i.top + i.bottom,
                h = o.helper.createSymbol(e, 0, 0, t[0], t[1]),
                l = Math.max(a, u);
            h.position = [i.left, i.top], a > u ? h.position[1] += (l - u) / 2 : h.position[0] += (l - a) / 2;
            var c = h.getBoundingRect();
            return h.position[0] -= c.x, h.position[1] -= c.y, h.setStyle(r), h.update(), h.__size = l, h
        }

        function a(e, t, r) {
            function n(e) { return e < 128 ? 1 : -1 }
            for (var i = t.width, a = t.height, o = e.canvas.width, s = e.canvas.height, u = i / o, h = a / s, l = e.createImageData(o, s), c = 0; c < s; c++)
                for (var d = 0; d < o; d++) {
                    var f = function(e, o) {
                            var s = 1 / 0;
                            e = Math.floor(e * u), o = Math.floor(o * h);
                            for (var l = o * i + e, c = t.data[4 * l], d = n(c), f = Math.max(o - r, 0); f < Math.min(o + r, a); f++)
                                for (var p = Math.max(e - r, 0); p < Math.min(e + r, i); p++) {
                                    var l = f * i + p,
                                        _ = t.data[4 * l],
                                        m = n(_),
                                        g = p - e,
                                        v = f - o;
                                    if (d !== m) {
                                        var y = g * g + v * v;
                                        y < s && (s = y)
                                    }
                                }
                            return d * Math.sqrt(s)
                        }(d, c),
                        p = f / r * .5 + .5,
                        _ = 4 * (c * o + d);
                    l.data[_++] = 255 * (1 - p), l.data[_++] = 255 * (1 - p), l.data[_++] = 255 * (1 - p), l.data[_++] = 255
                }
            return l
        }
        var o = r(0),
            s = {
                getMarginByStyle: function(e) {
                    var t = e.minMargin || 0,
                        r = 0;
                    e.stroke && "none" !== e.stroke && (r = null == e.lineWidth ? 1 : e.lineWidth);
                    var n = e.shadowBlur || 0,
                        i = e.shadowOffsetX || 0,
                        a = e.shadowOffsetY || 0,
                        o = {};
                    return o.left = Math.max(r / 2, -i + n, t), o.right = Math.max(r / 2, i + n, t), o.top = Math.max(r / 2, -a + n, t), o.bottom = Math.max(r / 2, a + n, t), o
                },
                createSymbolSprite: function(e, t, r, a) {
                    var o = i(e, t, r),
                        u = s.getMarginByStyle(r);
                    return { image: n(o.__size, a, function(e) { o.brush(e) }), margin: u }
                },
                createSDFFromCanvas: function(e, t, r, i) {
                    return n(t, i, function(t) {
                        var n = e.getContext("2d"),
                            i = n.getImageData(0, 0, e.width, e.height);
                        t.putImageData(a(t, i, r), 0, 0)
                    })
                },
                createSimpleSprite: function(e, t) {
                    return n(e, t, function(t) {
                        var r = e / 2;
                        t.beginPath(), t.arc(r, r, 60, 0, 2 * Math.PI, !1), t.closePath();
                        var n = t.createRadialGradient(r, r, 0, r, r, r);
                        n.addColorStop(0, "rgba(255, 255, 255, 1)"), n.addColorStop(.5, "rgba(255, 255, 255, 0.5)"), n.addColorStop(1, "rgba(255, 255, 255, 0)"), t.fillStyle = n, t.fill()
                    })
                }
            };
        e.exports = s
    }, function(e, t) {
        function r(e) { return e.valueOf() / y - .5 + x }

        function n(e) { return r(e) - T }

        function i(e, t) { return g(f(e) * p(b) - _(t) * f(b), p(e)) }

        function a(e, t) { return m(f(t) * p(b) + p(t) * f(b) * f(e)) }

        function o(e, t, r) { return g(f(e), p(e) * f(t) - _(r) * p(t)) }

        function s(e, t, r) { return m(f(t) * f(r) + p(t) * p(r) * p(e)) }

        function u(e, t) { return v * (280.16 + 360.9856235 * e) - t }

        function h(e) { return v * (357.5291 + .98560028 * e) }

        function l(e) { return e + v * (1.9148 * f(e) + .02 * f(2 * e) + 3e-4 * f(3 * e)) + 102.9372 * v + d }

        function c(e) {
            var t = h(e),
                r = l(t);
            return { dec: a(r, 0), ra: i(r, 0) }
        }
        var d = Math.PI,
            f = Math.sin,
            p = Math.cos,
            _ = Math.tan,
            m = Math.asin,
            g = Math.atan2,
            v = d / 180,
            y = 864e5,
            x = 2440588,
            T = 2451545,
            b = 23.4397 * v,
            w = {};
        w.getPosition = function(e, t, r) {
            var i = v * -r,
                a = v * t,
                h = n(e),
                l = c(h),
                d = u(h, i) - l.ra;
            return { azimuth: o(d, a, l.dec), altitude: s(d, a, l.dec) }
        }, e.exports = w
    }, function(e, t, r) {
        "use strict";

        function n(e) { return this._axes[e] }
        var i = r(15),
            a = function(e) { this._axes = {}, this._dimList = [], this.name = e || "" };
        a.prototype = {
            constructor: a,
            type: "cartesian",
            getAxis: function(e) { return this._axes[e] },
            getAxes: function() { return i.map(this._dimList, n, this) },
            getAxesByScale: function(e) { return e = e.toLowerCase(), i.filter(this.getAxes(), function(t) { return t.scale.type === e }) },
            addAxis: function(e) {
                var t = e.dim;
                this._axes[t] = e, this._dimList.push(t)
            },
            dataToCoord: function(e) { return this._dataCoordConvert(e, "dataToCoord") },
            coordToData: function(e) { return this._dataCoordConvert(e, "coordToData") },
            _dataCoordConvert: function(e, t) {
                for (var r = this._dimList, n = e instanceof Array ? [] : {}, i = 0; i < r.length; i++) {
                    var a = r[i],
                        o = this._axes[a];
                    n[a] = o[t](e[a])
                }
                return n
            }
        }, e.exports = a
    }, function(e, t, r) {
        var n = r(15),
            i = { Russia: [100, 60], "United States": [-99, 38], "United States of America": [-99, 38] };
        e.exports = function(e) {
            n.each(e.regions, function(e) {
                var t = i[e.name];
                if (t) {
                    var r = e.center;
                    r[0] = t[0], r[1] = t[1]
                }
            })
        }
    }, function(e, t, r) {
        var n = r(15),
            i = { "å—æµ·è¯¸å²›": [32, 80], "å¹¿ä¸œ": [0, -10], "é¦™æ¸¯": [10, 5], "æ¾³é—¨": [-10, 10], "å¤©æ´¥": [5, 5] };
        e.exports = function(e) {
            n.each(e.regions, function(e) {
                var t = i[e.name];
                if (t) {
                    var r = e.center;
                    r[0] += t[0] / 10.5, r[1] += -t[1] / 14
                }
            })
        }
    }, function(e, t, r) {
        "use strict";

        function n(e) { return "_EC_" + e }

        function i(e, t) { this.id = null == e ? "" : e, this.inEdges = [], this.outEdges = [], this.edges = [], this.hostGraph, this.dataIndex = null == t ? -1 : t }

        function a(e, t, r) { this.node1 = e, this.node2 = t, this.dataIndex = null == r ? -1 : r }
        var o = r(15),
            s = function(e) { this._directed = e || !1, this.nodes = [], this.edges = [], this._nodesMap = {}, this._edgesMap = {}, this.data, this.edgeData },
            u = s.prototype;
        u.type = "graph", u.isDirected = function() { return this._directed }, u.addNode = function(e, t) { e = e || "" + t; var r = this._nodesMap; if (!r[n(e)]) { var a = new i(e, t); return a.hostGraph = this, this.nodes.push(a), r[n(e)] = a, a } }, u.getNodeByIndex = function(e) { var t = this.data.getRawIndex(e); return this.nodes[t] }, u.getNodeById = function(e) { return this._nodesMap[n(e)] }, u.addEdge = function(e, t, r) {
            var o = this._nodesMap,
                s = this._edgesMap;
            if ("number" == typeof e && (e = this.nodes[e]), "number" == typeof t && (t = this.nodes[t]), e instanceof i || (e = o[n(e)]), t instanceof i || (t = o[n(t)]), e && t) { var u = e.id + "-" + t.id; if (!s[u]) { var h = new a(e, t, r); return h.hostGraph = this, this._directed && (e.outEdges.push(h), t.inEdges.push(h)), e.edges.push(h), e !== t && t.edges.push(h), this.edges.push(h), s[u] = h, h } }
        }, u.getEdgeByIndex = function(e) { var t = this.edgeData.getRawIndex(e); return this.edges[t] }, u.getEdge = function(e, t) { e instanceof i && (e = e.id), t instanceof i && (t = t.id); var r = this._edgesMap; return this._directed ? r[e + "-" + t] : r[e + "-" + t] || r[t + "-" + e] }, u.eachNode = function(e, t) { for (var r = this.nodes, n = r.length, i = 0; i < n; i++) r[i].dataIndex >= 0 && e.call(t, r[i], i) }, u.eachEdge = function(e, t) { for (var r = this.edges, n = r.length, i = 0; i < n; i++) r[i].dataIndex >= 0 && r[i].node1.dataIndex >= 0 && r[i].node2.dataIndex >= 0 && e.call(t, r[i], i) }, u.breadthFirstTraverse = function(e, t, r, a) {
            if (t instanceof i || (t = this._nodesMap[n(t)]), t) {
                for (var o = "out" === r ? "outEdges" : "in" === r ? "inEdges" : "edges", s = 0; s < this.nodes.length; s++) this.nodes[s].__visited = !1;
                if (!e.call(a, t, null))
                    for (var u = [t]; u.length;)
                        for (var h = u.shift(), l = h[o], s = 0; s < l.length; s++) {
                            var c = l[s],
                                d = c.node1 === h ? c.node2 : c.node1;
                            if (!d.__visited) {
                                if (e.call(a, d, h)) return;
                                u.push(d), d.__visited = !0
                            }
                        }
            }
        }, u.update = function() {
            for (var e = this.data, t = this.edgeData, r = this.nodes, n = this.edges, i = 0, a = r.length; i < a; i++) r[i].dataIndex = -1;
            for (var i = 0, a = e.count(); i < a; i++) r[e.getRawIndex(i)].dataIndex = i;
            t.filterSelf(function(e) { var r = n[t.getRawIndex(e)]; return r.node1.dataIndex >= 0 && r.node2.dataIndex >= 0 });
            for (var i = 0, a = n.length; i < a; i++) n[i].dataIndex = -1;
            for (var i = 0, a = t.count(); i < a; i++) n[t.getRawIndex(i)].dataIndex = i
        }, u.clone = function() {
            for (var e = new s(this._directed), t = this.nodes, r = this.edges, n = 0; n < t.length; n++) e.addNode(t[n].id, t[n].dataIndex);
            for (var n = 0; n < r.length; n++) {
                var i = r[n];
                e.addEdge(i.node1.id, i.node2.id, i.dataIndex)
            }
            return e
        }, i.prototype = { constructor: i, degree: function() { return this.edges.length }, inDegree: function() { return this.inEdges.length }, outDegree: function() { return this.outEdges.length }, getModel: function(e) { if (!(this.dataIndex < 0)) { return this.hostGraph.data.getItemModel(this.dataIndex).getModel(e) } } }, a.prototype.getModel = function(e) { if (!(this.dataIndex < 0)) { return this.hostGraph.edgeData.getItemModel(this.dataIndex).getModel(e) } };
        var h = function(e, t) { return { getValue: function(r) { var n = this[e][t]; return n.get(n.getDimension(r || "value"), this.dataIndex) }, setVisual: function(r, n) { this.dataIndex >= 0 && this[e][t].setItemVisual(this.dataIndex, r, n) }, getVisual: function(r, n) { return this[e][t].getItemVisual(this.dataIndex, r, n) }, setLayout: function(r, n) { this.dataIndex >= 0 && this[e][t].setItemLayout(this.dataIndex, r, n) }, getLayout: function() { return this[e][t].getItemLayout(this.dataIndex) }, getGraphicEl: function() { return this[e][t].getItemGraphicEl(this.dataIndex) }, getRawIndex: function() { return this[e][t].getRawIndex(this.dataIndex) } } };
        o.mixin(i, h("hostGraph", "data")), o.mixin(a, h("hostGraph", "edgeData")), s.Node = i, s.Edge = a, e.exports = s
    }, function(e, t, r) {
        function n(e) {
            var t = e.mainData,
                r = e.datas;
            r || (r = { main: t }, e.datasAttr = { main: "data" }), e.datas = e.mainData = null, h(t, r, e), d(r, function(r) { d(t.TRANSFERABLE_METHODS, function(t) { r.wrapMethod(t, c.curry(i, e)) }) }), t.wrapMethod("cloneShallow", c.curry(o, e)), d(t.CHANGABLE_METHODS, function(r) { t.wrapMethod(r, c.curry(a, e)) }), c.assert(r[t.dataType] === t)
        }

        function i(e, t) {
            if (u(this)) {
                var r = c.extend({}, this[f]);
                r[this.dataType] = t, h(t, r, e)
            } else l(t, this.dataType, this[p], e);
            return t
        }

        function a(e, t) { return e.struct && e.struct.update(this), t }

        function o(e, t) { return d(t[f], function(r, n) { r !== t && l(r.cloneShallow(), n, t, e) }), t }

        function s(e) { var t = this[p]; return null == e || null == t ? t : t[f][e] }

        function u(e) { return e[p] === e }

        function h(e, t, r) { e[f] = {}, d(t, function(t, n) { l(t, n, e, r) }) }

        function l(e, t, r, n) { r[f][t] = e, e[p] = r, e.dataType = t, n.struct && (e[n.structAttr] = n.struct, n.struct[n.datasAttr[t]] = e), e.getLinkedData = s }
        var c = r(15),
            d = c.each,
            f = "\0__link_datas",
            p = "\0__link_mainData";
        e.exports = n
    }, function(e, t, r) {
        var n = r(15),
            i = r(68),
            a = r(240),
            o = {};
        o.addCommas = function(e) { return isNaN(e) ? "-" : (e = (e + "").split("."), e[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, "$1,") + (e.length > 1 ? "." + e[1] : "")) }, o.toCamelCase = function(e, t) { return e = (e || "").toLowerCase().replace(/-(.)/g, function(e, t) { return t.toUpperCase() }), t && e && (e = e.charAt(0).toUpperCase() + e.slice(1)), e }, o.normalizeCssArray = n.normalizeCssArray;
        var s = o.encodeHTML = function(e) { return String(e).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;") },
            u = ["a", "b", "c", "d", "e", "f", "g"],
            h = function(e, t) { return "{" + e + (null == t ? "" : t) + "}" };
        o.formatTpl = function(e, t, r) {
            n.isArray(t) || (t = [t]);
            var i = t.length;
            if (!i) return "";
            for (var a = t[0].$vars || [], o = 0; o < a.length; o++) {
                var l = u[o],
                    c = h(l, 0);
                e = e.replace(h(l), r ? s(c) : c)
            }
            for (var d = 0; d < i; d++)
                for (var f = 0; f < a.length; f++) {
                    var c = t[d][a[f]];
                    e = e.replace(h(u[f], d), r ? s(c) : c)
                }
            return e
        }, o.formatTplSimple = function(e, t, r) { return n.each(t, function(t, n) { e = e.replace("{" + n + "}", r ? s(t) : t) }), e }, o.getTooltipMarker = function(e, t) { return e ? '<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:' + o.encodeHTML(e) + ";" + (t || "") + '"></span>' : "" };
        var l = function(e) { return e < 10 ? "0" + e : e };
        o.formatTime = function(e, t, r) {
            "week" !== e && "month" !== e && "quarter" !== e && "half-year" !== e && "year" !== e || (e = "MM-dd\nyyyy");
            var n = i.parseDate(t),
                a = r ? "UTC" : "",
                o = n["get" + a + "FullYear"](),
                s = n["get" + a + "Month"]() + 1,
                u = n["get" + a + "Date"](),
                h = n["get" + a + "Hours"](),
                c = n["get" + a + "Minutes"](),
                d = n["get" + a + "Seconds"]();
            return e = e.replace("MM", l(s)).replace("M", s).replace("yyyy", o).replace("yy", o % 100).replace("dd", l(u)).replace("d", u).replace("hh", l(h)).replace("h", h).replace("mm", l(c)).replace("m", c).replace("ss", l(d)).replace("s", d)
        }, o.capitalFirst = function(e) { return e ? e.charAt(0).toUpperCase() + e.substr(1) : e }, o.truncateText = a.truncateText, o.getTextRect = a.getBoundingRect, e.exports = o
    }, function(e, t, r) {
        "use strict";

        function n(e) { return { byte: d.Int8Array, ubyte: d.Uint8Array, short: d.Int16Array, ushort: d.Uint16Array }[e] || d.Float32Array }

        function i(e, t, r, n) { this.name = e, this.type = t, this.size = r, n && (this.semantic = n) }

        function a(e, t, r, n) {
            switch (i.call(this, e, t, r, n), this.value = null, r) {
                case 1:
                    this.get = function(e) { return this.value[e] }, this.set = function(e, t) { this.value[e] = t }, this.copy = function(e, t) { this.value[e] = this.value[e] };
                    break;
                case 2:
                    this.get = function(e, t) { var r = this.value; return t[0] = r[2 * e], t[1] = r[2 * e + 1], t }, this.set = function(e, t) {
                        var r = this.value;
                        r[2 * e] = t[0], r[2 * e + 1] = t[1]
                    }, this.copy = function(e, t) {
                        var r = this.value;
                        t *= 2, e *= 2, r[e] = r[t], r[e + 1] = r[t + 1]
                    };
                    break;
                case 3:
                    this.get = function(e, t) {
                        var r = 3 * e,
                            n = this.value;
                        return t[0] = n[r], t[1] = n[r + 1], t[2] = n[r + 2], t
                    }, this.set = function(e, t) {
                        var r = 3 * e,
                            n = this.value;
                        n[r] = t[0], n[r + 1] = t[1], n[r + 2] = t[2]
                    }, this.copy = function(e, t) {
                        var r = this.value;
                        t *= 3, e *= 3, r[e] = r[t], r[e + 1] = r[t + 1], r[e + 2] = r[t + 2]
                    };
                    break;
                case 4:
                    this.get = function(e, t) {
                        var r = this.value,
                            n = 4 * e;
                        return t[0] = r[n], t[1] = r[n + 1], t[2] = r[n + 2], t[3] = r[n + 3], t
                    }, this.set = function(e, t) {
                        var r = this.value,
                            n = 4 * e;
                        r[n] = t[0], r[n + 1] = t[1], r[n + 2] = t[2], r[n + 3] = t[3]
                    }, this.copy = function(e, t) {
                        var r = this.value;
                        t *= 4, e *= 4, r[e] = r[t], r[e + 1] = r[t + 1], r[e + 2] = r[t + 2], r[e + 3] = r[t + 3]
                    }
            }
        }

        function o(e, t, r, n, i) { this.name = e, this.type = t, this.buffer = r, this.size = n, this.semantic = i, this.symbol = "", this.needsRemove = !1 }

        function s(e) { this.buffer = e, this.count = 0 }

        function u() { console.warn("Geometry doesn't implement this method, use StaticGeometry instead") }
        var h = r(8),
            l = r(11),
            c = r(45),
            d = r(20);
        i.prototype.clone = function(e) { var t = new this.constructor(this.name, this.type, this.size, this.semantic); return e && console.warn("todo"), t }, a.prototype.constructor = new i, a.prototype.init = function(e) {
            if (!this.value || this.value.length != e * this.size) {
                var t = n(this.type);
                this.value = new t(e * this.size)
            }
        }, a.prototype.fromArray = function(e) {
            var t, r = n(this.type);
            if (e[0] && e[0].length) {
                var i = 0,
                    a = this.size;
                t = new r(e.length * a);
                for (var o = 0; o < e.length; o++)
                    for (var s = 0; s < a; s++) t[i++] = e[o][s]
            } else t = new r(e);
            this.value = t
        };
        var f = h.extend({ boundingBox: null, attributes: {}, indices: null, dynamic: !1 }, function() { this._cache = new c, this._attributeList = Object.keys(this.attributes) }, { pickByRay: null, pick: null, mainAttribute: "position", dirty: u, createAttribute: u, removeAttribute: u, getTriangleIndices: u, setTriangleIndices: u, isUseIndices: u, getEnabledAttributes: u, getBufferChunks: u, generateVertexNormals: u, generateFaceNormals: u, isUniqueVertex: u, generateUniqueVertex: u, generateTangents: u, generateBarycentric: u, applyTransform: u, dispose: u });
        f.STATIC_DRAW = l.STATIC_DRAW, f.DYNAMIC_DRAW = l.DYNAMIC_DRAW, f.STREAM_DRAW = l.STREAM_DRAW, f.AttributeBuffer = o, f.IndicesBuffer = s, f.Attribute = i, f.StaticAttribute = a, e.exports = f
    }, function(e, t, r) {
        "use strict";
        var n = r(12),
            i = r(37),
            a = i.extend(function() { return { name: "", inputs: {}, outputs: null, shader: "", inputLinks: {}, outputLinks: {}, pass: null, _prevOutputTextures: {}, _outputTextures: {}, _outputReferences: {}, _rendering: !1, _rendered: !1, _compositor: null } }, function() {
                var e = new n({ fragment: this.shader });
                this.pass = e
            }, {
                render: function(e, t) {
                    this.trigger("beforerender", e), this._rendering = !0;
                    var r = e.gl;
                    for (var n in this.inputLinks) {
                        var i = this.inputLinks[n],
                            a = i.node.getOutput(e, i.pin);
                        this.pass.setUniform(n, a)
                    }
                    if (this.outputs) {
                        this.pass.outputs = {};
                        var o = {};
                        for (var s in this.outputs) {
                            var u = this.updateParameter(s, e);
                            isNaN(u.width) && this.updateParameter(s, e);
                            var h = this.outputs[s],
                                l = this._compositor.allocateTexture(u);
                            this._outputTextures[s] = l;
                            var c = h.attachment || r.COLOR_ATTACHMENT0;
                            "string" == typeof c && (c = r[c]), o[c] = l
                        }
                        this._compositor.getFrameBuffer().bind(e);
                        for (var c in o) this._compositor.getFrameBuffer().attach(o[c], c);
                        this.pass.render(e), this._compositor.getFrameBuffer().updateMipmap(e.gl)
                    } else this.pass.outputs = null, this._compositor.getFrameBuffer().unbind(e), this.pass.render(e, t);
                    for (var n in this.inputLinks) {
                        var i = this.inputLinks[n];
                        i.node.removeReference(i.pin)
                    }
                    this._rendering = !1, this._rendered = !0, this.trigger("afterrender", e)
                },
                updateParameter: function(e, t) {
                    var r = this.outputs[e],
                        n = r.parameters,
                        i = r._parametersCopy;
                    if (i || (i = r._parametersCopy = {}), n)
                        for (var a in n) "width" !== a && "height" !== a && (i[a] = n[a]);
                    var o, s;
                    return o = n.width instanceof Function ? n.width.call(this, t) : n.width, s = n.height instanceof Function ? n.height.call(this, t) : n.height, i.width === o && i.height === s || this._outputTextures[e] && this._outputTextures[e].dispose(t.gl), i.width = o, i.height = s, i
                },
                setParameter: function(e, t) { this.pass.setUniform(e, t) },
                getParameter: function(e) { return this.pass.getUniform(e) },
                setParameters: function(e) { for (var t in e) this.setParameter(t, e[t]) },
                setShader: function(e) {
                    var t = this.pass.material;
                    t.shader.setFragment(e), t.attachShader(t.shader, !0)
                },
                shaderDefine: function(e, t) { this.pass.material.shader.define("fragment", e, t) },
                shaderUndefine: function(e) { this.pass.material.shader.undefine("fragment", e) },
                removeReference: function(e) { if (0 === --this._outputReferences[e]) { this.outputs[e].keepLastFrame ? (this._prevOutputTextures[e] && this._compositor.releaseTexture(this._prevOutputTextures[e]), this._prevOutputTextures[e] = this._outputTextures[e]) : this._compositor.releaseTexture(this._outputTextures[e]) } },
                link: function(e, t, r) { this.inputLinks[e] = { node: t, pin: r }, t.outputLinks[r] || (t.outputLinks[r] = []), t.outputLinks[r].push({ node: this, pin: e }), this.pass.material.shader.enableTexture(e) },
                clear: function() { i.prototype.clear.call(this), this.pass.material.shader.disableTexturesAll() },
                updateReference: function(e) {
                    if (!this._rendering) {
                        this._rendering = !0;
                        for (var t in this.inputLinks) {
                            var r = this.inputLinks[t];
                            r.node.updateReference(r.pin)
                        }
                        this._rendering = !1
                    }
                    e && this._outputReferences[e]++
                },
                beforeFrame: function() { this._rendered = !1; for (var e in this.outputLinks) this._outputReferences[e] = 0 },
                afterFrame: function() {
                    for (var e in this.outputLinks)
                        if (this._outputReferences[e] > 0) {
                            var t = this.outputs[e];
                            t.keepLastFrame ? (this._prevOutputTextures[e] && this._compositor.releaseTexture(this._prevOutputTextures[e]), this._prevOutputTextures[e] = this._outputTextures[e]) : this._compositor.releaseTexture(this._outputTextures[e])
                        }
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(8),
            i = r(37),
            a = n.extend(function() { return { nodes: [] } }, {
                dirty: function() { this._dirty = !0 },
                addNode: function(e) { this.nodes.indexOf(e) >= 0 || (this.nodes.push(e), this._dirty = !0) },
                removeNode: function(e) {
                    "string" == typeof e && (e = this.getNodeByName(e));
                    var t = this.nodes.indexOf(e);
                    t >= 0 && (this.nodes.splice(t, 1), this._dirty = !0)
                },
                getNodeByName: function(e) {
                    for (var t = 0; t < this.nodes.length; t++)
                        if (this.nodes[t].name === e) return this.nodes[t]
                },
                update: function() {
                    for (var e = 0; e < this.nodes.length; e++) this.nodes[e].clear();
                    for (var e = 0; e < this.nodes.length; e++) {
                        var t = this.nodes[e];
                        if (t.inputs)
                            for (var r in t.inputs)
                                if (t.inputs[r])
                                    if (!t.pass || t.pass.material.isUniformEnabled(r)) {
                                        var n = t.inputs[r],
                                            i = this.findPin(n);
                                        i ? t.link(r, i.node, i.pin) : "string" == typeof n ? console.warn("Node " + n + " not exist") : console.warn("Pin of " + n.node + "." + n.pin + " not exist")
                                    } else console.warn("Pin " + t.name + "." + r + " not used.")
                    }
                },
                findPin: function(e) {
                    var t;
                    if (("string" == typeof e || e instanceof i) && (e = { node: e }), "string" == typeof e.node)
                        for (var r = 0; r < this.nodes.length; r++) {
                            var n = this.nodes[r];
                            n.name === e.node && (t = n)
                        } else t = e.node;
                    if (t) { var a = e.pin; if (a || t.outputs && (a = Object.keys(t.outputs)[0]), t.outputs[a]) return { node: t, pin: a } }
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(37),
            i = r(17),
            a = r(11),
            o = r(10),
            s = n.extend({ name: "scene", scene: null, camera: null, autoUpdateScene: !0, preZ: !1 }, function() { this.frameBuffer = new o }, {
                render: function(e) {
                    this._rendering = !0;
                    var t = e.gl;
                    this.trigger("beforerender");
                    var r;
                    if (this.outputs) {
                        var n = this.frameBuffer;
                        for (var o in this.outputs) {
                            var s = this.updateParameter(o, e),
                                u = this.outputs[o],
                                h = this._compositor.allocateTexture(s);
                            this._outputTextures[o] = h;
                            var l = u.attachment || t.COLOR_ATTACHMENT0;
                            "string" == typeof l && (l = t[l]), n.attach(h, l)
                        }
                        n.bind(e);
                        var c = i.getExtension(t, "EXT_draw_buffers");
                        if (c) {
                            var d = [];
                            for (var l in this.outputs)(l = parseInt(l)) >= t.COLOR_ATTACHMENT0 && l <= t.COLOR_ATTACHMENT0 + 8 && d.push(l);
                            c.drawBuffersEXT(d)
                        }
                        e.saveClear(), e.clearBit = a.DEPTH_BUFFER_BIT | a.COLOR_BUFFER_BIT, r = e.render(this.scene, this.camera, !this.autoUpdateScene, this.preZ), e.restoreClear(), n.unbind(e)
                    } else r = e.render(this.scene, this.camera, !this.autoUpdateScene, this.preZ);
                    this.trigger("afterrender", r), this._rendering = !1, this._rendered = !0
                }
            });
        e.exports = s
    }, function(e, t, r) {
        "use strict";
        var n = r(37),
            i = n.extend(function() { return { texture: null, outputs: { color: {} } } }, function() {}, { getOutput: function(e, t) { return this.texture }, beforeFrame: function() {}, afterFrame: function() {} });
        e.exports = i
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r) {
            "object" == typeof t && (r = t, t = null);
            var n, o = this;
            if (!(e instanceof Function)) { n = []; for (var s in e) e.hasOwnProperty(s) && n.push(s) }
            var u = function(t) {
                if (o.apply(this, arguments), e instanceof Function ? i(this, e.call(this, t)) : a(this, e, n), this.constructor === u)
                    for (var r = u.__initializers__, s = 0; s < r.length; s++) r[s].apply(this, arguments)
            };
            u.__super__ = o, o.__initializers__ ? u.__initializers__ = o.__initializers__.slice() : u.__initializers__ = [], t && u.__initializers__.push(t);
            var h = function() {};
            return h.prototype = o.prototype, u.prototype = new h, u.prototype.constructor = u, i(u.prototype, r), u.extend = o.extend, u.derive = o.extend, u
        }

        function i(e, t) {
            if (t)
                for (var r in t) t.hasOwnProperty(r) && (e[r] = t[r])
        }

        function a(e, t, r) {
            for (var n = 0; n < r.length; n++) {
                var i = r[n];
                e[i] = t[i]
            }
        }
        e.exports = { extend: n, derive: n }
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = n.extend({ castShadow: !1 }, {
                type: "AMBIENT_LIGHT",
                uniformTemplates: {
                    ambientLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    }
                }
            });
        e.exports = i
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = r(228),
            a = n.extend({ cubemap: null, castShadow: !1, _normalDistribution: null, _brdfLookup: null }, {
                type: "AMBIENT_CUBEMAP_LIGHT",
                prefilter: function(e, t) {
                    this._brdfLookup || (this._normalDistribution = i.generateNormalDistribution(), this._brdfLookup = i.integrateBRDF(e, this._normalDistribution));
                    var r = this.cubemap;
                    if (!r.__prefiltered) {
                        var n = i.prefilterEnvironmentMap(e, r, { encodeRGBM: !0, width: t, height: t }, this._normalDistribution, this._brdfLookup);
                        this.cubemap = n.environmentMap, this.cubemap.__prefiltered = !0, r.dispose(e.gl)
                    }
                },
                uniformTemplates: {
                    ambientCubemapLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    },
                    ambientCubemapLightCubemap: { type: "t", value: function(e) { return e.cubemap } },
                    ambientCubemapLightBRDFLookup: { type: "t", value: function(e) { return e._brdfLookup } }
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(19),
            i = r(20),
            a = n.extend({ castShadow: !1, coefficients: [] }, function() { this._coefficientsTmpArr = new i.Float32Array(27) }, {
                type: "AMBIENT_SH_LIGHT",
                uniformTemplates: {
                    ambientSHLightColor: {
                        type: "3f",
                        value: function(e) {
                            var t = e.color,
                                r = e.intensity;
                            return [t[0] * r, t[1] * r, t[2] * r]
                        }
                    },
                    ambientSHLightCoefficients: { type: "3f", value: function(e) { for (var t = e._coefficientsTmpArr, r = 0; r < e.coefficients.length; r++) t[r] = e.coefficients[r]; return t } }
                }
            });
        e.exports = a
    }, function(e, t, r) {
        "use strict";

        function n(e, t) {
            return function(r) {
                var n = r.getDevicePixelRatio(),
                    i = r.getWidth(),
                    a = r.getHeight(),
                    o = t(i, a, n);
                this.setParameter(e, o)
            }
        }

        function i(e, t) {
            return function(e) {
                var r = e.getDevicePixelRatio(),
                    n = e.getWidth(),
                    i = e.getHeight();
                return t(n, i, r)
            }
        }

        function a(e) { var t = /^expr\((.*)\)$/.exec(e); if (t) try { var r = new Function("width", "height", "dpr", "return " + t[1]); return r(1, 1), r } catch (e) { throw new Error("Invalid expression.") } }
        var o = r(8),
            s = r(73),
            u = r(27),
            h = r(71),
            l = (r(37), r(199)),
            c = r(200),
            d = r(197),
            f = r(7),
            p = r(6),
            _ = r(5),
            m = r(23),
            g = /#source\((.*?)\)/,
            v = /#url\((.*?)\)/,
            y = o.extend({ rootPath: "", textureRootPath: "", shaderRootPath: "", scene: null, camera: null }, {
                load: function(e) {
                    var t = this;
                    this.rootPath || (this.rootPath = e.slice(0, e.lastIndexOf("/"))), s.get({ url: e, onprogress: function(e, r, n) { t.trigger("progress", e, r, n) }, onerror: function(e) { t.trigger("error", e) }, responseType: "text", onload: function(e) { t.parse(JSON.parse(e)) } })
                },
                parse: function(e) {
                    var t = this,
                        r = new h,
                        n = { textures: {}, shaders: {}, parameters: {} },
                        i = function(i, a) {
                            for (var o = 0; o < e.nodes.length; o++) {
                                var s = e.nodes[o],
                                    u = t._createNode(s, n);
                                u && r.addNode(u)
                            }
                            t.trigger("success", r)
                        };
                    for (var a in e.parameters) {
                        var o = e.parameters[a];
                        n.parameters[a] = this._convertParameter(o)
                    }
                    return this._loadShaders(e, function(r) { t._loadTextures(e, n, function(e) { n.textures = e, n.shaders = r, i() }) }), r
                },
                _createNode: function(e, t) {
                    var r, i, o, s = e.type || "filter";
                    if ("filter" === s) {
                        var u = e.shader.trim(),
                            h = g.exec(u);
                        if (h ? r = f.source(h[1].trim()) : "#" === u.charAt(0) && (r = t.shaders[u.substr(1)]), r || (r = u), !r) return
                    }
                    if (e.inputs) { i = {}; for (var p in e.inputs) "string" == typeof e.inputs[p] ? i[p] = e.inputs[p] : i[p] = { node: e.inputs[p].node, pin: e.inputs[p].pin } }
                    if (e.outputs) { o = {}; for (var p in e.outputs) { var _ = e.outputs[p]; if (o[p] = {}, null != _.attachment && (o[p].attachment = _.attachment), null != _.keepLastFrame && (o[p].keepLastFrame = _.keepLastFrame), null != _.outputLastFrame && (o[p].outputLastFrame = _.outputLastFrame), "string" == typeof _.parameters) { var m = _.parameters; "#" === m.charAt(0) && (o[p].parameters = t.parameters[m.substr(1)]) } else _.parameters && (o[p].parameters = this._convertParameter(_.parameters)) } }
                    var v;
                    if (v = "scene" === s ? new l({ name: e.name, scene: this.scene, camera: this.camera, outputs: o }) : "texture" === s ? new c({ name: e.name, outputs: o }) : new d({ name: e.name, shader: r, inputs: i, outputs: o })) {
                        if (e.parameters)
                            for (var p in e.parameters) { var y = e.parameters[p]; "string" == typeof y && (y = y.trim(), "#" === y.charAt(0) ? y = t.textures[y.substr(1)] : v.on("beforerender", n(p, a(y)))), v.setParameter(p, y) }
                        if (e.defines && v.pass)
                            for (var p in e.defines) {
                                var y = e.defines[p];
                                v.pass.material.shader.define("fragment", p, y)
                            }
                    }
                    return v
                },
                _convertParameter: function(e) {
                    var t = {};
                    return e ? (["type", "minFilter", "magFilter", "wrapS", "wrapT", "flipY", "useMipmap"].forEach(function(r) {
                        var n = e[r];
                        null != n && ("string" == typeof n && (n = p[n]), t[r] = n)
                    }), ["width", "height"].forEach(function(r) { if (null != e[r]) { var n = e[r]; "string" == typeof n ? (n = n.trim(), t[r] = i(r, a(n))) : t[r] = n } }), null != e.useMipmap && (t.useMipmap = e.useMipmap), t) : t
                },
                _loadShaders: function(e, t) {
                    if (!e.shaders) return void t({});
                    var r = {},
                        n = 0,
                        i = !1,
                        a = this.shaderRootPath || this.rootPath;
                    u.each(e.shaders, function(e, o) {
                        var h = v.exec(e);
                        if (h) {
                            var l = h[1];
                            l = u.relative2absolute(l, a), n++, s.get({ url: l, onload: function(e) { r[o] = e, f.import(e), 0 === --n && (t(r), i = !0) } })
                        } else r[o] = e, f.import(e)
                    }, this), 0 !== n || i || t(r)
                },
                _loadTextures: function(e, t, r) {
                    if (!e.textures) return void r({});
                    var n = {},
                        i = 0,
                        a = !1,
                        o = this.textureRootPath || this.rootPath;
                    u.each(e.textures, function(e, t) {
                        var s, h = e.path,
                            l = this._convertParameter(e.parameters);
                        if (h instanceof Array && 6 === h.length) h = h.map(function(e) { return u.relative2absolute(e, o) }), s = new m(l);
                        else {
                            if ("string" != typeof h) return;
                            h = u.relative2absolute(h, o), s = new _(l)
                        }
                        s.load(h), i++, s.once("success", function() { n[t] = s, 0 === --i && (r(n), a = !0) })
                    }, this), 0 !== i || a || r(n)
                }
            });
        e.exports = y
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.mat2,
            a = function() { this._array = i.create(), this._dirty = !0 };
        a.prototype = { constructor: a, setArray: function(e) { for (var t = 0; t < this._array.length; t++) this._array[t] = e[t]; return this._dirty = !0, this }, clone: function() { return (new a).copy(this) }, copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this }, adjoint: function() { return i.adjoint(this._array, this._array), this._dirty = !0, this }, determinant: function() { return i.determinant(this._array) }, identity: function() { return i.identity(this._array), this._dirty = !0, this }, invert: function() { return i.invert(this._array, this._array), this._dirty = !0, this }, mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this }, mulLeft: function(e) { return i.mul(this._array, e._array, this._array), this._dirty = !0, this }, multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this }, multiplyLeft: function(e) { return i.multiply(this._array, e._array, this._array), this._dirty = !0, this }, rotate: function(e) { return i.rotate(this._array, this._array, e), this._dirty = !0, this }, scale: function(e) { return i.scale(this._array, this._array, e._array), this._dirty = !0, this }, transpose: function() { return i.transpose(this._array, this._array), this._dirty = !0, this }, toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" }, toArray: function() { return Array.prototype.slice.call(this._array) } }, a.adjoint = function(e, t) { return i.adjoint(e._array, t._array), e._dirty = !0, e }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.determinant = function(e) { return i.determinant(e._array) }, a.identity = function(e) { return i.identity(e._array), e._dirty = !0, e }, a.invert = function(e, t) { return i.invert(e._array, t._array), e._dirty = !0, e }, a.mul = function(e, t, r) { return i.mul(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.rotate = function(e, t, r) { return i.rotate(e._array, t._array, r), e._dirty = !0, e }, a.scale = function(e, t, r) { return i.scale(e._array, t._array, r._array), e._dirty = !0, e }, a.transpose = function(e, t) { return i.transpose(e._array, t._array), e._dirty = !0, e }, e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.mat2d,
            a = function() { this._array = i.create(), this._dirty = !0 };
        a.prototype = { constructor: a, setArray: function(e) { for (var t = 0; t < this._array.length; t++) this._array[t] = e[t]; return this._dirty = !0, this }, clone: function() { return (new a).copy(this) }, copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this }, determinant: function() { return i.determinant(this._array) }, identity: function() { return i.identity(this._array), this._dirty = !0, this }, invert: function() { return i.invert(this._array, this._array), this._dirty = !0, this }, mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this }, mulLeft: function(e) { return i.mul(this._array, e._array, this._array), this._dirty = !0, this }, multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this }, multiplyLeft: function(e) { return i.multiply(this._array, e._array, this._array), this._dirty = !0, this }, rotate: function(e) { return i.rotate(this._array, this._array, e), this._dirty = !0, this }, scale: function(e) { return i.scale(this._array, this._array, e._array), this._dirty = !0, this }, translate: function(e) { return i.translate(this._array, this._array, e._array), this._dirty = !0, this }, toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" }, toArray: function() { return Array.prototype.slice.call(this._array) } }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.determinant = function(e) { return i.determinant(e._array) }, a.identity = function(e) { return i.identity(e._array), e._dirty = !0, e }, a.invert = function(e, t) { return i.invert(e._array, t._array), e._dirty = !0, e }, a.mul = function(e, t, r) { return i.mul(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.rotate = function(e, t, r) { return i.rotate(e._array, t._array, r), e._dirty = !0, e }, a.scale = function(e, t, r) { return i.scale(e._array, t._array, r._array), e._dirty = !0, e }, a.translate = function(e, t, r) { return i.translate(e._array, t._array, r._array), e._dirty = !0, e }, e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.mat3,
            a = function() { this._array = i.create(), this._dirty = !0 };
        a.prototype = { constructor: a, setArray: function(e) { for (var t = 0; t < this._array.length; t++) this._array[t] = e[t]; return this._dirty = !0, this }, adjoint: function() { return i.adjoint(this._array, this._array), this._dirty = !0, this }, clone: function() { return (new a).copy(this) }, copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this }, determinant: function() { return i.determinant(this._array) }, fromMat2d: function(e) { return i.fromMat2d(this._array, e._array), this._dirty = !0, this }, fromMat4: function(e) { return i.fromMat4(this._array, e._array), this._dirty = !0, this }, fromQuat: function(e) { return i.fromQuat(this._array, e._array), this._dirty = !0, this }, identity: function() { return i.identity(this._array), this._dirty = !0, this }, invert: function() { return i.invert(this._array, this._array), this._dirty = !0, this }, mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this }, mulLeft: function(e) { return i.mul(this._array, e._array, this._array), this._dirty = !0, this }, multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this }, multiplyLeft: function(e) { return i.multiply(this._array, e._array, this._array), this._dirty = !0, this }, rotate: function(e) { return i.rotate(this._array, this._array, e), this._dirty = !0, this }, scale: function(e) { return i.scale(this._array, this._array, e._array), this._dirty = !0, this }, translate: function(e) { return i.translate(this._array, this._array, e._array), this._dirty = !0, this }, normalFromMat4: function(e) { return i.normalFromMat4(this._array, e._array), this._dirty = !0, this }, transpose: function() { return i.transpose(this._array, this._array), this._dirty = !0, this }, toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" }, toArray: function() { return Array.prototype.slice.call(this._array) } }, a.adjoint = function(e, t) { return i.adjoint(e._array, t._array), e._dirty = !0, e }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.determinant = function(e) { return i.determinant(e._array) }, a.identity = function(e) { return i.identity(e._array), e._dirty = !0, e }, a.invert = function(e, t) { return i.invert(e._array, t._array), e }, a.mul = function(e, t, r) { return i.mul(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.fromMat2d = function(e, t) { return i.fromMat2d(e._array, t._array), e._dirty = !0, e }, a.fromMat4 = function(e, t) { return i.fromMat4(e._array, t._array), e._dirty = !0, e }, a.fromQuat = function(e, t) { return i.fromQuat(e._array, t._array), e._dirty = !0, e }, a.normalFromMat4 = function(e, t) { return i.normalFromMat4(e._array, t._array), e._dirty = !0, e }, a.rotate = function(e, t, r) { return i.rotate(e._array, t._array, r), e._dirty = !0, e }, a.scale = function(e, t, r) { return i.scale(e._array, t._array, r._array), e._dirty = !0, e }, a.transpose = function(e, t) { return i.transpose(e._array, t._array), e._dirty = !0, e }, a.translate = function(e, t, r) { return i.translate(e._array, t._array, r._array), e._dirty = !0, e }, e.exports = a
    }, function(e, t, r) {
        "use strict";
        var n = r(1),
            i = n.vec4,
            a = function(e, t, r, n) { e = e || 0, t = t || 0, r = r || 0, n = n || 0, this._array = i.fromValues(e, t, r, n), this._dirty = !0 };
        a.prototype = { constructor: a, add: function(e) { return i.add(this._array, this._array, e._array), this._dirty = !0, this }, set: function(e, t, r, n) { return this._array[0] = e, this._array[1] = t, this._array[2] = r, this._array[3] = n, this._dirty = !0, this }, setArray: function(e) { return this._array[0] = e[0], this._array[1] = e[1], this._array[2] = e[2], this._array[3] = e[3], this._dirty = !0, this }, clone: function() { return new a(this.x, this.y, this.z, this.w) }, copy: function(e) { return i.copy(this._array, e._array), this._dirty = !0, this }, dist: function(e) { return i.dist(this._array, e._array) }, distance: function(e) { return i.distance(this._array, e._array) }, div: function(e) { return i.div(this._array, this._array, e._array), this._dirty = !0, this }, divide: function(e) { return i.divide(this._array, this._array, e._array), this._dirty = !0, this }, dot: function(e) { return i.dot(this._array, e._array) }, len: function() { return i.len(this._array) }, length: function() { return i.length(this._array) }, lerp: function(e, t, r) { return i.lerp(this._array, e._array, t._array, r), this._dirty = !0, this }, min: function(e) { return i.min(this._array, this._array, e._array), this._dirty = !0, this }, max: function(e) { return i.max(this._array, this._array, e._array), this._dirty = !0, this }, mul: function(e) { return i.mul(this._array, this._array, e._array), this._dirty = !0, this }, multiply: function(e) { return i.multiply(this._array, this._array, e._array), this._dirty = !0, this }, negate: function() { return i.negate(this._array, this._array), this._dirty = !0, this }, normalize: function() { return i.normalize(this._array, this._array), this._dirty = !0, this }, random: function(e) { return i.random(this._array, e), this._dirty = !0, this }, scale: function(e) { return i.scale(this._array, this._array, e), this._dirty = !0, this }, scaleAndAdd: function(e, t) { return i.scaleAndAdd(this._array, this._array, e._array, t), this._dirty = !0, this }, sqrDist: function(e) { return i.sqrDist(this._array, e._array) }, squaredDistance: function(e) { return i.squaredDistance(this._array, e._array) }, sqrLen: function() { return i.sqrLen(this._array) }, squaredLength: function() { return i.squaredLength(this._array) }, sub: function(e) { return i.sub(this._array, this._array, e._array), this._dirty = !0, this }, subtract: function(e) { return i.subtract(this._array, this._array, e._array), this._dirty = !0, this }, transformMat4: function(e) { return i.transformMat4(this._array, this._array, e._array), this._dirty = !0, this }, transformQuat: function(e) { return i.transformQuat(this._array, this._array, e._array), this._dirty = !0, this }, toString: function() { return "[" + Array.prototype.join.call(this._array, ",") + "]" }, toArray: function() { return Array.prototype.slice.call(this._array) } };
        var o = Object.defineProperty;
        if (o) {
            var s = a.prototype;
            o(s, "x", { get: function() { return this._array[0] }, set: function(e) { this._array[0] = e, this._dirty = !0 } }), o(s, "y", { get: function() { return this._array[1] }, set: function(e) { this._array[1] = e, this._dirty = !0 } }), o(s, "z", { get: function() { return this._array[2] }, set: function(e) { this._array[2] = e, this._dirty = !0 } }), o(s, "w", { get: function() { return this._array[3] }, set: function(e) { this._array[3] = e, this._dirty = !0 } })
        }
        a.add = function(e, t, r) { return i.add(e._array, t._array, r._array), e._dirty = !0, e }, a.set = function(e, t, r, n, a) { i.set(e._array, t, r, n, a), e._dirty = !0 }, a.copy = function(e, t) { return i.copy(e._array, t._array), e._dirty = !0, e }, a.dist = function(e, t) { return i.distance(e._array, t._array) }, a.distance = a.dist, a.div = function(e, t, r) { return i.divide(e._array, t._array, r._array), e._dirty = !0, e }, a.divide = a.div, a.dot = function(e, t) { return i.dot(e._array, t._array) }, a.len = function(e) { return i.length(e._array) }, a.lerp = function(e, t, r, n) { return i.lerp(e._array, t._array, r._array, n), e._dirty = !0, e }, a.min = function(e, t, r) { return i.min(e._array, t._array, r._array), e._dirty = !0, e }, a.max = function(e, t, r) { return i.max(e._array, t._array, r._array), e._dirty = !0, e }, a.mul = function(e, t, r) { return i.multiply(e._array, t._array, r._array), e._dirty = !0, e }, a.multiply = a.mul, a.negate = function(e, t) { return i.negate(e._array, t._array), e._dirty = !0, e }, a.normalize = function(e, t) { return i.normalize(e._array, t._array), e._dirty = !0, e }, a.random = function(e, t) { return i.random(e._array, t), e._dirty = !0, e }, a.scale = function(e, t, r) { return i.scale(e._array, t._array, r), e._dirty = !0, e }, a.scaleAndAdd = function(e, t, r, n) { return i.scaleAndAdd(e._array, t._array, r._array, n), e._dirty = !0, e }, a.sqrDist = function(e, t) { return i.sqrDist(e._array, t._array) }, a.squaredDistance = a.sqrDist, a.sqrLen = function(e) { return i.sqrLen(e._array) }, a.squaredLength = a.sqrLen, a.sub = function(e, t, r) { return i.subtract(e._array, t._array, r._array), e._dirty = !0, e }, a.subtract = a.sub, a.transformMat4 = function(e, t, r) { return i.transformMat4(e._array, t._array, r._array), e._dirty = !0, e }, a.transformQuat = function(e, t, r) { return i.transformQuat(e._array, t._array, r._array), e._dirty = !0, e }, e.exports = a
    }, function(e, t, r) {
        var n = r(8),
            i = r(56),
            a = r(28),
            o = r(3),
            s = r(9),
            u = r(70),
            h = r(11),
            l = n.extend({ scene: null, camera: null, renderer: null }, function() { this._ray = new i, this._ndc = new a }, {
                pick: function(e, t, r) { return this.pickAll(e, t, [], r)[0] || null },
                pickAll: function(e, t, r, n) { return this.renderer.screenToNDC(e, t, this._ndc), this.camera.castRay(this._ndc, this._ray), r = r || [], this._intersectNode(this.scene, r, n || !1), r.sort(this._intersectionCompareFunc), r },
                _intersectNode: function(e, t, r) { e instanceof u && e.isRenderable() && (e.ignorePicking && !r || !(e.mode === h.TRIANGLES && e.geometry.isUseIndices() || e.geometry.pickByRay || e.geometry.pick) || this._intersectRenderable(e, t)); for (var n = 0; n < e._children.length; n++) this._intersectNode(e._children[n], t, r) },
                _intersectRenderable: function() {
                    var e = new o,
                        t = new o,
                        r = new o,
                        n = new i,
                        a = new s;
                    return function(i, u) {
                        n.copy(this._ray), s.invert(a, i.worldTransform), n.applyTransform(a);
                        var c = i.geometry;
                        if (!c.boundingBox || n.intersectBoundingBox(c.boundingBox)) {
                            if (c.pick) return void c.pick(this._ndc.x, this._ndc.y, this.renderer, this.camera, i, u);
                            if (c.pickByRay) return void c.pickByRay(n, i, u);
                            var d, f = i.cullFace === h.BACK && i.frontFace === h.CCW || i.cullFace === h.FRONT && i.frontFace === h.CW,
                                p = c.indices,
                                _ = c.attributes.position;
                            if (_ && _.value && p)
                                for (var m = 0; m < p.length; m += 3) {
                                    var g = p[m],
                                        v = p[m + 1],
                                        y = p[m + 2];
                                    if (_.get(g, e._array), _.get(v, t._array), _.get(y, r._array), d = f ? n.intersectTriangle(e, t, r, i.culling) : n.intersectTriangle(e, r, t, i.culling)) {
                                        var x = new o;
                                        o.transformMat4(x, d, i.worldTransform), u.push(new l.Intersection(d, x, i, [g, v, y], m / 3, o.dist(x, this._ray.origin)))
                                    }
                                }
                        }
                    }
                }(),
                _intersectionCompareFunc: function(e, t) { return e.distance - t.distance }
            });
        l.Intersection = function(e, t, r, n, i, a) { this.point = e, this.pointWorld = t, this.target = r, this.triangle = n, this.triangleIndex = i, this.distance = a }, e.exports = l
    }, function(e, t, r) {
        var n = r(8),
            i = r(11),
            a = r(3),
            o = r(14),
            s = r(54),
            u = r(9),
            h = r(52),
            l = r(7),
            c = (r(19), r(25), r(78)),
            d = r(76),
            f = r(77),
            p = (r(81), r(16)),
            _ = r(10),
            m = r(6),
            g = r(5),
            v = r(23),
            y = r(44),
            x = r(36),
            T = r(12),
            b = r(72),
            w = r(1),
            E = w.mat4,
            S = (w.vec3, ["px", "nx", "py", "ny", "pz", "nz"]);
        l.import(r(225));
        var A = n.extend(function() { return { softShadow: A.PCF, shadowBlur: 1, lightFrustumBias: 2, kernelPCF: new Float32Array([1, 0, 1, 1, -1, 1, 0, 1, -1, 0, -1, -1, 1, -1, 0, -1]), precision: "mediump", _frameBuffer: new _, _textures: {}, _shadowMapNumber: { POINT_LIGHT: 0, DIRECTIONAL_LIGHT: 0, SPOT_LIGHT: 0 }, _meshMaterials: {}, _depthMaterials: {}, _depthShaders: {}, _distanceMaterials: {}, _opaqueCasters: [], _receivers: [], _lightsCastShadow: [], _lightCameras: {}, _texturePool: new b } }, function() { this._gaussianPassH = new T({ fragment: l.source("qtek.compositor.gaussian_blur") }), this._gaussianPassV = new T({ fragment: l.source("qtek.compositor.gaussian_blur") }), this._gaussianPassH.setUniform("blurSize", this.shadowBlur), this._gaussianPassH.setUniform("blurDir", 0), this._gaussianPassV.setUniform("blurSize", this.shadowBlur), this._gaussianPassV.setUniform("blurDir", 1), this._outputDepthPass = new T({ fragment: l.source("qtek.sm.debug_depth") }) }, {
            render: function(e, t, r, n) { this.trigger("beforerender", this, e, t, r), this._renderShadowPass(e, t, r, n), this.trigger("afterrender", this, e, t, r) },
            renderDebug: function(e, t) {
                e.saveClear();
                var r = e.viewport,
                    n = 0,
                    i = t || r.width / 4,
                    a = i;
                this.softShadow === A.VSM ? this._outputDepthPass.material.shader.define("fragment", "USE_VSM") : this._outputDepthPass.material.shader.undefine("fragment", "USE_VSM");
                for (var o in this._textures) {
                    var s = this._textures[o];
                    e.setViewport(n, 0, i * s.width / s.height, a), this._outputDepthPass.setUniform("depthMap", s), this._outputDepthPass.render(e), n += i * s.width / s.height
                }
                e.setViewport(r), e.restoreClear()
            },
            _bindDepthMaterial: function(e, t, r) {
                for (var n = 0; n < e.length; n++) {
                    var i, a, o = e[n],
                        s = o.material.shadowTransparentMap instanceof g,
                        u = o.material.shadowTransparentMap,
                        h = o.joints && o.joints.length;
                    s ? (i = h + "-" + u.__GUID__, a = h + "-t") : (i = h, a = h), o.useSkinMatricesTexture && (i += "-s", a += "-s");
                    var c = o.shadowDepthMaterial || this._depthMaterials[i],
                        d = o.shadowDepthMaterial ? o.shadowDepthMaterial.shader : this._depthShaders[a];
                    o.material !== c && (d || (d = new l({ vertex: l.source("qtek.sm.depth.vertex"), fragment: l.source("qtek.sm.depth.fragment"), precision: this.precision }), h > 0 && (d.define("vertex", "SKINNING"), d.define("vertex", "JOINT_COUNT", h)), s && d.define("both", "SHADOW_TRANSPARENT"), o.useSkinMatricesTexture && d.define("vertex", "USE_SKIN_MATRICES_TEXTURE"), this._depthShaders[a] = d), c || (c = new p({ shader: d }), this._depthMaterials[i] = c), o.material = c, this.softShadow === A.VSM ? d.define("fragment", "USE_VSM") : d.undefine("fragment", "USE_VSM"), c.setUniform("bias", t), c.setUniform("slopeScale", r), s && c.set("shadowTransparentMap", u))
                }
            },
            _bindDistanceMaterial: function(e, t) {
                for (var r = t.getWorldPosition()._array, n = 0; n < e.length; n++) {
                    var i = e[n],
                        a = i.joints && i.joints.length,
                        o = this._distanceMaterials[a];
                    i.material !== o && (o || (o = new p({ shader: new l({ vertex: l.source("qtek.sm.distance.vertex"), fragment: l.source("qtek.sm.distance.fragment"), precision: this.precision }) }), a > 0 && (o.shader.define("vertex", "SKINNING"), o.shader.define("vertex", "JOINT_COUNT", a)), this._distanceMaterials[a] = o), i.material = o, this.softShadow === A.VSM ? o.shader.define("fragment", "USE_VSM") : o.shader.undefine("fragment", "USE_VSM")), o.set("lightPosition", r), o.set("range", t.range)
                }
            },
            saveMaterial: function(e) {
                for (var t = 0; t < e.length; t++) {
                    var r = e[t];
                    this._meshMaterials[r.__GUID__] = r.material
                }
            },
            restoreMaterial: function(e) {
                for (var t = 0; t < e.length; t++) {
                    var r = e[t],
                        n = this._meshMaterials[r.__GUID__];
                    n && (r.material = n)
                }
            },
            _updateCasterAndReceiver: function(e, t) {
                t.castShadow && this._opaqueCasters.push(t), t.receiveShadow ? (this._receivers.push(t), t.material.set("shadowEnabled", 1), t.material.set("pcfKernel", this.kernelPCF)) : t.material.set("shadowEnabled", 0), !t.material.shader && t.material.updateShader && t.material.updateShader(e.gl);
                var r = t.material.shader;
                if (this.softShadow === A.VSM) r.define("fragment", "USE_VSM"), r.undefine("fragment", "PCF_KERNEL_SIZE");
                else {
                    r.undefine("fragment", "USE_VSM");
                    var n = this.kernelPCF;
                    n && n.length ? r.define("fragment", "PCF_KERNEL_SIZE", n.length / 2) : r.undefine("fragment", "PCF_KERNEL_SIZE")
                }
            },
            _update: function(e, t) {
                for (var r = 0; r < t.opaqueQueue.length; r++) this._updateCasterAndReceiver(e, t.opaqueQueue[r]);
                for (var r = 0; r < t.transparentQueue.length; r++) this._updateCasterAndReceiver(e, t.transparentQueue[r]);
                for (var r = 0; r < t.lights.length; r++) {
                    var n = t.lights[r];
                    n.castShadow && this._lightsCastShadow.push(n)
                }
            },
            _renderShadowPass: function(e, t, r, n) {
                function i(e) { return e.height }
                for (var a in this._shadowMapNumber) this._shadowMapNumber[a] = 0;
                this._lightsCastShadow.length = 0, this._opaqueCasters.length = 0, this._receivers.length = 0;
                var o = e.gl;
                if (n || t.update(), this._update(e, t), this._lightsCastShadow.length) {
                    o.enable(o.DEPTH_TEST), o.depthMask(!0), o.disable(o.BLEND), o.clearColor(1, 1, 1, 1);
                    var s = [],
                        u = [],
                        h = [],
                        l = [],
                        p = [],
                        _ = [];
                    this.saveMaterial(this._opaqueCasters);
                    for (var m, g = 0; g < this._lightsCastShadow.length; g++) {
                        var v = this._lightsCastShadow[g];
                        if (v instanceof d) {
                            if (m) { console.warn("Only one dire light supported with shadow cascade"); continue }
                            if (v.shadowCascade > 1 && (m = v, v.shadowCascade > 4)) { console.warn("Support at most 4 cascade"); continue }
                            this.renderDirectionalLightShadow(e, t, r, v, this._opaqueCasters, p, l, h)
                        } else v instanceof c ? this.renderSpotLightShadow(e, v, this._opaqueCasters, u, s) : v instanceof f && this.renderPointLightShadow(e, v, this._opaqueCasters, _);
                        this._shadowMapNumber[v.type]++
                    }
                    this.restoreMaterial(this._opaqueCasters);
                    var y = p.slice(),
                        x = p.slice();
                    y.pop(), x.shift(), y.reverse(), x.reverse(), l.reverse();
                    for (var T = s.map(i), b = h.map(i), w = {}, g = 0; g < this._receivers.length; g++) {
                        var E = this._receivers[g],
                            S = E.material,
                            A = S.shader;
                        if (!w[A.__GUID__]) {
                            var M = !1;
                            for (var N in this._shadowMapNumber) {
                                var C = this._shadowMapNumber[N],
                                    L = N + "_SHADOWMAP_COUNT";
                                A.fragmentDefines[L] !== C && C > 0 && (A.fragmentDefines[L] = C, M = !0)
                            }
                            M && A.dirty(), m ? A.define("fragment", "SHADOW_CASCADE", m.shadowCascade) : A.undefine("fragment", "SHADOW_CASCADE"), w[A.__GUID__] = !0
                        }
                        s.length > 0 && (S.setUniform("spotLightShadowMaps", s), S.setUniform("spotLightMatrices", u), S.setUniform("spotLightShadowMapSizes", T)), h.length > 0 && (S.setUniform("directionalLightShadowMaps", h), m && (S.setUniform("shadowCascadeClipsNear", y), S.setUniform("shadowCascadeClipsFar", x)), S.setUniform("directionalLightMatrices", l), S.setUniform("directionalLightShadowMapSizes", b)), _.length > 0 && S.setUniform("pointLightShadowMaps", _)
                    }
                }
            },
            renderDirectionalLightShadow: function() {
                var e = new s,
                    t = new u,
                    r = new o,
                    n = new u,
                    i = new u,
                    a = new u,
                    l = new u;
                return function(o, s, c, d, f, p, _, m) {
                    var g = d.shadowBias;
                    this._bindDepthMaterial(f, g, d.shadowSlopeScale), f.sort(h.opaqueSortFunc);
                    var v = Math.min(-s.viewBoundingBoxLastFrame.min.z, c.far),
                        x = Math.max(-s.viewBoundingBoxLastFrame.max.z, c.near),
                        T = this._getDirectionalLightCamera(d, s, c),
                        b = a._array;
                    l.copy(T.projectionMatrix), E.invert(i._array, T.worldTransform._array), E.multiply(i._array, i._array, c.worldTransform._array), E.multiply(b, l._array, i._array);
                    for (var w = [], S = c instanceof y, M = (c.near + c.far) / (c.near - c.far), N = 2 * c.near * c.far / (c.near - c.far), C = 0; C <= d.shadowCascade; C++) {
                        var L = x * Math.pow(v / x, C / d.shadowCascade),
                            D = x + (v - x) * C / d.shadowCascade,
                            I = L * d.cascadeSplitLogFactor + D * (1 - d.cascadeSplitLogFactor);
                        w.push(I), p.push(-(-I * M + N) / -I)
                    }
                    var R = this._getTexture(d, d.shadowCascade);
                    m.push(R);
                    var P = o.viewport,
                        O = o.gl;
                    this._frameBuffer.attach(R), this._frameBuffer.bind(o), O.clear(O.COLOR_BUFFER_BIT | O.DEPTH_BUFFER_BIT);
                    for (var C = 0; C < d.shadowCascade; C++) {
                        var F = w[C],
                            B = w[C + 1];
                        S ? E.perspective(t._array, c.fov / 180 * Math.PI, c.aspect, F, B) : E.ortho(t._array, c.left, c.right, c.bottom, c.top, F, B), e.setFromProjection(t), e.getTransformedBoundingBox(r, i), r.applyProjection(l);
                        var U = r.min._array,
                            z = r.max._array;
                        n.ortho(U[0], z[0], U[1], z[1], 1, -1), T.projectionMatrix.multiplyLeft(n);
                        var G = d.shadowResolution || 512;
                        o.setViewport((d.shadowCascade - C - 1) * G, 0, G, G, 1);
                        for (var k in this._depthMaterials) this._depthMaterials[k].set("shadowBias", g);
                        o.renderQueue(f, T), this.softShadow === A.VSM && this._gaussianFilter(o, R, R.width);
                        var H = new u;
                        H.copy(T.worldTransform).invert().multiplyLeft(T.projectionMatrix), _.push(H._array), T.projectionMatrix.copy(l)
                    }
                    this._frameBuffer.unbind(o), o.setViewport(P)
                }
            }(),
            renderSpotLightShadow: function(e, t, r, n, i) {
                this._bindDepthMaterial(r, t.shadowBias, t.shadowSlopeScale), r.sort(h.opaqueSortFunc);
                var a = this._getTexture(t),
                    o = this._getSpotLightCamera(t),
                    s = e.gl;
                this._frameBuffer.attach(a), this._frameBuffer.bind(e), s.clear(s.COLOR_BUFFER_BIT | s.DEPTH_BUFFER_BIT), e.renderQueue(r, o), this._frameBuffer.unbind(e), this.softShadow === A.VSM && this._gaussianFilter(e, a, a.width);
                var l = new u;
                l.copy(o.worldTransform).invert().multiplyLeft(o.projectionMatrix), i.push(a), n.push(l._array)
            },
            renderPointLightShadow: function(e, t, r, n) {
                var i = this._getTexture(t),
                    a = e.gl;
                n.push(i), this._bindDistanceMaterial(r, t);
                for (var o = 0; o < 6; o++) {
                    var s = S[o],
                        u = this._getPointLightCamera(t, s);
                    this._frameBuffer.attach(i, a.COLOR_ATTACHMENT0, a.TEXTURE_CUBE_MAP_POSITIVE_X + o), this._frameBuffer.bind(e), a.clear(a.COLOR_BUFFER_BIT | a.DEPTH_BUFFER_BIT), e.renderQueue(r, u)
                }
                this._frameBuffer.unbind(e)
            },
            _gaussianFilter: function(e, t, r) {
                var n = { width: r, height: r, type: m.FLOAT },
                    i = (e.gl, this._texturePool.get(n));
                this._frameBuffer.attach(i), this._frameBuffer.bind(e), this._gaussianPassH.setUniform("texture", t), this._gaussianPassH.setUniform("textureWidth", r), this._gaussianPassH.render(e), this._frameBuffer.attach(t), this._gaussianPassV.setUniform("texture", i), this._gaussianPassV.setUniform("textureHeight", r), this._gaussianPassV.render(e), this._frameBuffer.unbind(e), this._texturePool.put(i)
            },
            _getTexture: function(e, t) {
                var r = e.__GUID__,
                    n = this._textures[r],
                    a = e.shadowResolution || 512;
                return t = t || 1, n || (n = e instanceof f ? new v : new g, n.width = a * t, n.height = a, this.softShadow === A.VSM ? (n.type = m.FLOAT, n.anisotropic = 4) : (n.minFilter = i.NEAREST, n.magFilter = i.NEAREST, n.useMipmap = !1), this._textures[r] = n), n
            },
            _getPointLightCamera: function(e, t) {
                this._lightCameras.point || (this._lightCameras.point = { px: new y, nx: new y, py: new y, ny: new y, pz: new y, nz: new y });
                var r = this._lightCameras.point[t];
                switch (r.far = e.range, r.fov = 90, r.position.set(0, 0, 0), t) {
                    case "px":
                        r.lookAt(a.POSITIVE_X, a.NEGATIVE_Y);
                        break;
                    case "nx":
                        r.lookAt(a.NEGATIVE_X, a.NEGATIVE_Y);
                        break;
                    case "py":
                        r.lookAt(a.POSITIVE_Y, a.POSITIVE_Z);
                        break;
                    case "ny":
                        r.lookAt(a.NEGATIVE_Y, a.NEGATIVE_Z);
                        break;
                    case "pz":
                        r.lookAt(a.POSITIVE_Z, a.NEGATIVE_Y);
                        break;
                    case "nz":
                        r.lookAt(a.NEGATIVE_Z, a.NEGATIVE_Y)
                }
                return e.getWorldPosition(r.position), r.update(), r
            },
            _getDirectionalLightCamera: function() {
                var e = new u,
                    t = new o,
                    r = new o;
                return function(n, i, a) {
                    this._lightCameras.directional || (this._lightCameras.directional = new x);
                    var o = this._lightCameras.directional;
                    t.copy(i.viewBoundingBoxLastFrame), t.intersection(a.frustum.boundingBox), o.position.copy(t.min).add(t.max).scale(.5).transformMat4(a.worldTransform), o.rotation.copy(n.rotation), o.scale.copy(n.scale), o.updateWorldTransform(), e.copy(o.worldTransform).invert().multiply(a.worldTransform), r.copy(t).applyTransform(e);
                    var s = r.min._array,
                        u = r.max._array;
                    return o.position.scaleAndAdd(o.worldTransform.z, u[2] + this.lightFrustumBias), o.near = 0, o.far = -s[2] + u[2] + this.lightFrustumBias, o.left = s[0] - this.lightFrustumBias, o.right = u[0] + this.lightFrustumBias, o.top = u[1] + this.lightFrustumBias, o.bottom = s[1] - this.lightFrustumBias, o.update(!0), o
                }
            }(),
            _getSpotLightCamera: function(e) { this._lightCameras.spot || (this._lightCameras.spot = new y); var t = this._lightCameras.spot; return t.fov = 2 * e.penumbraAngle, t.far = e.range, t.worldTransform.copy(e.worldTransform), t.updateProjectionMatrix(), E.invert(t.viewMatrix._array, t.worldTransform._array), t },
            dispose: function(e) {
                var t = e.gl || e;
                for (var r in this._depthMaterials) {
                    var n = this._depthMaterials[r];
                    n.dispose(t)
                }
                for (var r in this._distanceMaterials) {
                    var n = this._distanceMaterials[r];
                    n.dispose(t)
                }
                this._frameBuffer && this._frameBuffer.dispose(t);
                for (var i in this._textures) this._textures[i].dispose(t);
                this._texturePool.clear(e.gl), this._depthMaterials = {}, this._distanceMaterials = {}, this._textures = {}, this._lightCameras = {}, this._shadowMapNumber = { POINT_LIGHT: 0, DIRECTIONAL_LIGHT: 0, SPOT_LIGHT: 0 }, this._meshMaterials = {};
                for (var a = 0; a < this._receivers.length; a++) {
                    var o = this._receivers[a];
                    if (o.material && o.material.shader) {
                        var s = o.material,
                            u = s.shader;
                        u.undefine("fragment", "POINT_LIGHT_SHADOW_COUNT"), u.undefine("fragment", "DIRECTIONAL_LIGHT_SHADOW_COUNT"), u.undefine("fragment", "AMBIENT_LIGHT_SHADOW_COUNT"), s.set("shadowEnabled", 0)
                    }
                }
                this._opaqueCasters = [], this._receivers = [], this._lightsCastShadow = []
            }
        });
        A.VSM = 1, A.PCF = 2, e.exports = A
    }, function(e, t) { e.exports = "@export qtek.basic.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nuniform vec2 uvRepeat : [1.0, 1.0];\nuniform vec2 uvOffset : [0.0, 0.0];\n\nattribute vec2 texcoord : TEXCOORD_0;\nattribute vec3 position : POSITION;\n\nattribute vec3 barycentric;\n\n@import qtek.chunk.skinning_header\n\nvarying vec2 v_Texcoord;\nvarying vec3 v_Barycentric;\n\nvoid main()\n{\n vec3 skinnedPosition = position;\n\n#ifdef SKINNING\n @import qtek.chunk.skin_matrix\n\n skinnedPosition = (skinMatrixWS * vec4(position, 1.0)).xyz;\n#endif\n\n v_Texcoord = texcoord * uvRepeat + uvOffset;\n v_Barycentric = barycentric;\n\n gl_Position = worldViewProjection * vec4(skinnedPosition, 1.0);\n}\n\n@end\n\n\n\n\n@export qtek.basic.fragment\n\n\nvarying vec2 v_Texcoord;\nuniform sampler2D diffuseMap;\nuniform vec3 color : [1.0, 1.0, 1.0];\nuniform vec3 emission : [0.0, 0.0, 0.0];\nuniform float alpha : 1.0;\n\n#ifdef ALPHA_TEST\nuniform float alphaCutoff: 0.9;\n#endif\n\nuniform float lineWidth : 0.0;\nuniform vec3 lineColor : [0.0, 0.0, 0.0];\nvarying vec3 v_Barycentric;\n\n@import qtek.util.edge_factor\n\n@import qtek.util.rgbm\n\n@import qtek.util.srgb\n\nvoid main()\n{\n\n#ifdef RENDER_TEXCOORD\n gl_FragColor = vec4(v_Texcoord, 1.0, 1.0);\n return;\n#endif\n\n gl_FragColor = vec4(color, alpha);\n\n#ifdef DIFFUSEMAP_ENABLED\n vec4 tex = decodeHDR(texture2D(diffuseMap, v_Texcoord));\n\n#ifdef SRGB_DECODE\n tex = sRGBToLinear(tex);\n#endif\n\n#if defined(DIFFUSEMAP_ALPHA_ALPHA)\n gl_FragColor.a = tex.a;\n#endif\n\n gl_FragColor.rgb *= tex.rgb;\n#endif\n\n gl_FragColor.rgb += emission;\n if( lineWidth > 0.01)\n {\n gl_FragColor.rgb = gl_FragColor.rgb * mix(lineColor, vec3(1.0), edgeFactor(lineWidth));\n }\n\n#ifdef GAMMA_ENCODE\n gl_FragColor.rgb = pow(gl_FragColor.rgb, vec3(1 / 2.2));\n#endif\n\n#ifdef ALPHA_TEST\n if (gl_FragColor.a < alphaCutoff) {\n discard;\n }\n#endif\n\n gl_FragColor = encodeHDR(gl_FragColor);\n\n}\n\n@end" }, function(e, t) { e.exports = "@export qtek.compositor.blend\n#ifdef TEXTURE1_ENABLED\nuniform sampler2D texture1;\nuniform float weight1 : 1.0;\n#endif\n#ifdef TEXTURE2_ENABLED\nuniform sampler2D texture2;\nuniform float weight2 : 1.0;\n#endif\n#ifdef TEXTURE3_ENABLED\nuniform sampler2D texture3;\nuniform float weight3 : 1.0;\n#endif\n#ifdef TEXTURE4_ENABLED\nuniform sampler2D texture4;\nuniform float weight4 : 1.0;\n#endif\n#ifdef TEXTURE5_ENABLED\nuniform sampler2D texture5;\nuniform float weight5 : 1.0;\n#endif\n#ifdef TEXTURE6_ENABLED\nuniform sampler2D texture6;\nuniform float weight6 : 1.0;\n#endif\n\nvarying vec2 v_Texcoord;\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec4 tex = vec4(0.0);\n#ifdef TEXTURE1_ENABLED\n tex += decodeHDR(texture2D(texture1, v_Texcoord)) * weight1;\n#endif\n#ifdef TEXTURE2_ENABLED\n tex += decodeHDR(texture2D(texture2, v_Texcoord)) * weight2;\n#endif\n#ifdef TEXTURE3_ENABLED\n tex += decodeHDR(texture2D(texture3, v_Texcoord)) * weight3;\n#endif\n#ifdef TEXTURE4_ENABLED\n tex += decodeHDR(texture2D(texture4, v_Texcoord)) * weight4;\n#endif\n#ifdef TEXTURE5_ENABLED\n tex += decodeHDR(texture2D(texture5, v_Texcoord)) * weight5;\n#endif\n#ifdef TEXTURE6_ENABLED\n tex += decodeHDR(texture2D(texture6, v_Texcoord)) * weight6;\n#endif\n\n gl_FragColor = encodeHDR(tex);\n}\n@end" }, function(e, t) { e.exports = "@export qtek.compositor.kernel.gaussian_9\nfloat gaussianKernel[9];\ngaussianKernel[0] = 0.07;\ngaussianKernel[1] = 0.09;\ngaussianKernel[2] = 0.12;\ngaussianKernel[3] = 0.14;\ngaussianKernel[4] = 0.16;\ngaussianKernel[5] = 0.14;\ngaussianKernel[6] = 0.12;\ngaussianKernel[7] = 0.09;\ngaussianKernel[8] = 0.07;\n@end\n\n@export qtek.compositor.kernel.gaussian_13\n\nfloat gaussianKernel[13];\n\ngaussianKernel[0] = 0.02;\ngaussianKernel[1] = 0.03;\ngaussianKernel[2] = 0.06;\ngaussianKernel[3] = 0.08;\ngaussianKernel[4] = 0.11;\ngaussianKernel[5] = 0.13;\ngaussianKernel[6] = 0.14;\ngaussianKernel[7] = 0.13;\ngaussianKernel[8] = 0.11;\ngaussianKernel[9] = 0.08;\ngaussianKernel[10] = 0.06;\ngaussianKernel[11] = 0.03;\ngaussianKernel[12] = 0.02;\n\n@end\n\n\n@export qtek.compositor.gaussian_blur\n\n#define SHADER_NAME gaussian_blur\n\nuniform sampler2D texture; varying vec2 v_Texcoord;\n\nuniform float blurSize : 2.0;\nuniform vec2 textureSize : [512.0, 512.0];\nuniform float blurDir : 0.0;\n\n@import qtek.util.rgbm\n@import qtek.util.clamp_sample\n\nvoid main (void)\n{\n @import qtek.compositor.kernel.gaussian_9\n\n vec2 off = blurSize / textureSize;\n off *= vec2(1.0 - blurDir, blurDir);\n\n vec4 sum = vec4(0.0);\n float weightAll = 0.0;\n\n for (int i = 0; i < 9; i++) {\n float w = gaussianKernel[i];\n vec4 texel = decodeHDR(clampSample(texture, v_Texcoord + float(i - 4) * off));\n sum += texel * w;\n weightAll += w;\n }\n gl_FragColor = encodeHDR(sum / max(weightAll, 0.01));\n}\n\n@end\n" }, function(e, t) { e.exports = "@export qtek.compositor.bright\n\nuniform sampler2D texture;\n\nuniform float threshold : 1;\nuniform float scale : 1.0;\n\nuniform vec2 textureSize: [512, 512];\n\nvarying vec2 v_Texcoord;\n\nconst vec3 lumWeight = vec3(0.2125, 0.7154, 0.0721);\n\n@import qtek.util.rgbm\n\n\nvec4 median(vec4 a, vec4 b, vec4 c)\n{\n return a + b + c - min(min(a, b), c) - max(max(a, b), c);\n}\n\nvoid main()\n{\n vec4 texel = decodeHDR(texture2D(texture, v_Texcoord));\n\n#ifdef ANTI_FLICKER\n vec3 d = 1.0 / textureSize.xyx * vec3(1.0, 1.0, 0.0);\n\n vec4 s1 = decodeHDR(texture2D(texture, v_Texcoord - d.xz));\n vec4 s2 = decodeHDR(texture2D(texture, v_Texcoord + d.xz));\n vec4 s3 = decodeHDR(texture2D(texture, v_Texcoord - d.zy));\n vec4 s4 = decodeHDR(texture2D(texture, v_Texcoord + d.zy));\n texel = median(median(texel, s1, s2), s3, s4);\n\n#endif\n\n float lum = dot(texel.rgb , lumWeight);\n vec4 color;\n if (lum > threshold && texel.a > 0.0)\n {\n color = vec4(texel.rgb * scale, texel.a * scale);\n }\n else\n {\n color = vec4(0.0);\n }\n\n gl_FragColor = encodeHDR(color);\n}\n@end\n" }, function(e, t) { e.exports = "@export qtek.compositor.downsample\n\nuniform sampler2D texture;\nuniform vec2 textureSize : [512, 512];\n\nvarying vec2 v_Texcoord;\n\n@import qtek.util.rgbm\nfloat brightness(vec3 c)\n{\n return max(max(c.r, c.g), c.b);\n}\n\n@import qtek.util.clamp_sample\n\nvoid main()\n{\n vec4 d = vec4(-1.0, -1.0, 1.0, 1.0) / textureSize.xyxy;\n\n#ifdef ANTI_FLICKER\n vec3 s1 = decodeHDR(clampSample(texture, v_Texcoord + d.xy)).rgb;\n vec3 s2 = decodeHDR(clampSample(texture, v_Texcoord + d.zy)).rgb;\n vec3 s3 = decodeHDR(clampSample(texture, v_Texcoord + d.xw)).rgb;\n vec3 s4 = decodeHDR(clampSample(texture, v_Texcoord + d.zw)).rgb;\n\n float s1w = 1.0 / (brightness(s1) + 1.0);\n float s2w = 1.0 / (brightness(s2) + 1.0);\n float s3w = 1.0 / (brightness(s3) + 1.0);\n float s4w = 1.0 / (brightness(s4) + 1.0);\n float oneDivideSum = 1.0 / (s1w + s2w + s3w + s4w);\n\n vec4 color = vec4(\n (s1 * s1w + s2 * s2w + s3 * s3w + s4 * s4w) * oneDivideSum,\n 1.0\n );\n#else\n vec4 color = decodeHDR(clampSample(texture, v_Texcoord + d.xy));\n color += decodeHDR(clampSample(texture, v_Texcoord + d.zy));\n color += decodeHDR(clampSample(texture, v_Texcoord + d.xw));\n color += decodeHDR(clampSample(texture, v_Texcoord + d.zw));\n color *= 0.25;\n#endif\n\n gl_FragColor = encodeHDR(color);\n}\n\n@end" }, function(e, t) { e.exports = "@export qtek.compositor.fxaa\n\nuniform sampler2D texture;\nuniform vec4 viewport : VIEWPORT;\n\nvarying vec2 v_Texcoord;\n\n#define FXAA_REDUCE_MIN (1.0/128.0)\n#define FXAA_REDUCE_MUL (1.0/8.0)\n#define FXAA_SPAN_MAX 8.0\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec2 resolution = 1.0 / viewport.zw;\n vec3 rgbNW = decodeHDR( texture2D( texture, ( gl_FragCoord.xy + vec2( -1.0, -1.0 ) ) * resolution ) ).xyz;\n vec3 rgbNE = decodeHDR( texture2D( texture, ( gl_FragCoord.xy + vec2( 1.0, -1.0 ) ) * resolution ) ).xyz;\n vec3 rgbSW = decodeHDR( texture2D( texture, ( gl_FragCoord.xy + vec2( -1.0, 1.0 ) ) * resolution ) ).xyz;\n vec3 rgbSE = decodeHDR( texture2D( texture, ( gl_FragCoord.xy + vec2( 1.0, 1.0 ) ) * resolution ) ).xyz;\n vec4 rgbaM = decodeHDR( texture2D( texture, gl_FragCoord.xy * resolution ) );\n vec3 rgbM = rgbaM.xyz;\n float opacity = rgbaM.w;\n\n vec3 luma = vec3( 0.299, 0.587, 0.114 );\n\n float lumaNW = dot( rgbNW, luma );\n float lumaNE = dot( rgbNE, luma );\n float lumaSW = dot( rgbSW, luma );\n float lumaSE = dot( rgbSE, luma );\n float lumaM = dot( rgbM, luma );\n float lumaMin = min( lumaM, min( min( lumaNW, lumaNE ), min( lumaSW, lumaSE ) ) );\n float lumaMax = max( lumaM, max( max( lumaNW, lumaNE) , max( lumaSW, lumaSE ) ) );\n\n vec2 dir;\n dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));\n dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));\n\n float dirReduce = max( ( lumaNW + lumaNE + lumaSW + lumaSE ) * ( 0.25 * FXAA_REDUCE_MUL ), FXAA_REDUCE_MIN );\n\n float rcpDirMin = 1.0 / ( min( abs( dir.x ), abs( dir.y ) ) + dirReduce );\n dir = min( vec2( FXAA_SPAN_MAX, FXAA_SPAN_MAX),\n max( vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),\n dir * rcpDirMin)) * resolution;\n\n vec3 rgbA = decodeHDR( texture2D( texture, gl_FragCoord.xy * resolution + dir * ( 1.0 / 3.0 - 0.5 ) ) ).xyz;\n rgbA += decodeHDR( texture2D( texture, gl_FragCoord.xy * resolution + dir * ( 2.0 / 3.0 - 0.5 ) ) ).xyz;\n rgbA *= 0.5;\n\n vec3 rgbB = decodeHDR( texture2D( texture, gl_FragCoord.xy * resolution + dir * -0.5 ) ).xyz;\n rgbB += decodeHDR( texture2D( texture, gl_FragCoord.xy * resolution + dir * 0.5 ) ).xyz;\n rgbB *= 0.25;\n rgbB += rgbA * 0.5;\n\n float lumaB = dot( rgbB, luma );\n\n if ( ( lumaB < lumaMin ) || ( lumaB > lumaMax ) )\n {\n gl_FragColor = vec4( rgbA, opacity );\n\n }\n else {\n\n gl_FragColor = vec4( rgbB, opacity );\n\n }\n}\n\n@end" }, function(e, t) { e.exports = "@export qtek.compositor.hdr.composite\n\nuniform sampler2D texture;\n#ifdef BLOOM_ENABLED\nuniform sampler2D bloom;\n#endif\n#ifdef LENSFLARE_ENABLED\nuniform sampler2D lensflare;\nuniform sampler2D lensdirt;\n#endif\n\n#ifdef LUM_ENABLED\nuniform sampler2D lum;\n#endif\n\n#ifdef LUT_ENABLED\nuniform sampler2D lut;\n#endif\n\n#ifdef COLOR_CORRECTION\nuniform float brightness : 0.0;\nuniform float contrast : 1.0;\nuniform float saturation : 1.0;\n#endif\n\n#ifdef VIGNETTE\nuniform float vignetteDarkness: 1.0;\nuniform float vignetteOffset: 1.0;\n#endif\n\nuniform float exposure : 1.0;\nuniform float bloomIntensity : 0.25;\nuniform float lensflareIntensity : 1;\n\nvarying vec2 v_Texcoord;\n\n\n@import qtek.util.srgb\n\n\n\n\nvec3 ACESToneMapping(vec3 color)\n{\n const float A = 2.51;\n const float B = 0.03;\n const float C = 2.43;\n const float D = 0.59;\n const float E = 0.14;\n return (color * (A * color + B)) / (color * (C * color + D) + E);\n}\n\nfloat eyeAdaption(float fLum)\n{\n return mix(0.2, fLum, 0.5);\n}\n\n#ifdef LUT_ENABLED\nvec3 lutTransform(vec3 color) {\n float blueColor = color.b * 63.0;\n\n vec2 quad1;\n quad1.y = floor(floor(blueColor) / 8.0);\n quad1.x = floor(blueColor) - (quad1.y * 8.0);\n\n vec2 quad2;\n quad2.y = floor(ceil(blueColor) / 8.0);\n quad2.x = ceil(blueColor) - (quad2.y * 8.0);\n\n vec2 texPos1;\n texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);\n texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);\n\n vec2 texPos2;\n texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.r);\n texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * color.g);\n\n vec4 newColor1 = texture2D(lut, texPos1);\n vec4 newColor2 = texture2D(lut, texPos2);\n\n vec4 newColor = mix(newColor1, newColor2, fract(blueColor));\n return newColor.rgb;\n}\n#endif\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec4 texel = vec4(0.0);\n vec4 originalTexel = vec4(0.0);\n#ifdef TEXTURE_ENABLED\n texel = decodeHDR(texture2D(texture, v_Texcoord));\n originalTexel = texel;\n#endif\n\n#ifdef BLOOM_ENABLED\n vec4 bloomTexel = decodeHDR(texture2D(bloom, v_Texcoord));\n texel.rgb += bloomTexel.rgb * bloomIntensity;\n texel.a += bloomTexel.a * bloomIntensity;\n#endif\n\n#ifdef LENSFLARE_ENABLED\n texel += decodeHDR(texture2D(lensflare, v_Texcoord)) * texture2D(lensdirt, v_Texcoord) * lensflareIntensity;\n#endif\n\n texel.a = min(texel.a, 1.0);\n\n#ifdef LUM_ENABLED\n float fLum = texture2D(lum, vec2(0.5, 0.5)).r;\n float adaptedLumDest = 3.0 / (max(0.1, 1.0 + 10.0*eyeAdaption(fLum)));\n float exposureBias = adaptedLumDest * exposure;\n#else\n float exposureBias = exposure;\n#endif\n texel.rgb *= exposureBias;\n\n texel.rgb = ACESToneMapping(texel.rgb);\n texel = linearTosRGB(texel);\n\n#ifdef LUT_ENABLED\n texel.rgb = lutTransform(clamp(texel.rgb,vec3(0.0),vec3(1.0)));\n#endif\n\n#ifdef COLOR_CORRECTION\n texel.rgb = clamp(texel.rgb + vec3(brightness), 0.0, 1.0);\n texel.rgb = clamp((texel.rgb - vec3(0.5))*contrast+vec3(0.5), 0.0, 1.0);\n float lum = dot(texel.rgb, vec3(0.2125, 0.7154, 0.0721));\n texel.rgb = mix(vec3(lum), texel.rgb, saturation);\n#endif\n\n#ifdef VIGNETTE\n vec2 uv = (v_Texcoord - vec2(0.5)) * vec2(vignetteOffset);\n texel.rgb = mix(texel.rgb, vec3(1.0 - vignetteDarkness), dot(uv, uv));\n#endif\n\n gl_FragColor = encodeHDR(texel);\n\n#ifdef DEBUG\n #if DEBUG == 1\n gl_FragColor = encodeHDR(decodeHDR(texture2D(texture, v_Texcoord)));\n #elif DEBUG == 2\n gl_FragColor = encodeHDR(decodeHDR(texture2D(bloom, v_Texcoord)) * bloomIntensity);\n #elif DEBUG == 3\n gl_FragColor = encodeHDR(decodeHDR(texture2D(lensflare, v_Texcoord) * lensflareIntensity));\n #endif\n#endif\n\n if (originalTexel.a <= 0.01) {\n gl_FragColor.a = dot(gl_FragColor.rgb, vec3(0.2125, 0.7154, 0.0721));\n }\n #ifdef PREMULTIPLY_ALPHA\n gl_FragColor.rgb *= gl_FragColor.a;\n#endif\n}\n\n@end" }, function(e, t) { e.exports = "\n@export qtek.compositor.lut\n\nvarying vec2 v_Texcoord;\n\nuniform sampler2D texture;\nuniform sampler2D lookup;\n\nvoid main()\n{\n\n vec4 tex = texture2D(texture, v_Texcoord);\n\n float blueColor = tex.b * 63.0;\n\n vec2 quad1;\n quad1.y = floor(floor(blueColor) / 8.0);\n quad1.x = floor(blueColor) - (quad1.y * 8.0);\n\n vec2 quad2;\n quad2.y = floor(ceil(blueColor) / 8.0);\n quad2.x = ceil(blueColor) - (quad2.y * 8.0);\n\n vec2 texPos1;\n texPos1.x = (quad1.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * tex.r);\n texPos1.y = (quad1.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * tex.g);\n\n vec2 texPos2;\n texPos2.x = (quad2.x * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * tex.r);\n texPos2.y = (quad2.y * 0.125) + 0.5/512.0 + ((0.125 - 1.0/512.0) * tex.g);\n\n vec4 newColor1 = texture2D(lookup, texPos1);\n vec4 newColor2 = texture2D(lookup, texPos2);\n\n vec4 newColor = mix(newColor1, newColor2, fract(blueColor));\n gl_FragColor = vec4(newColor.rgb, tex.w);\n}\n\n@end" }, function(e, t) { e.exports = "@export qtek.compositor.output\n\n#define OUTPUT_ALPHA\n\nvarying vec2 v_Texcoord;\n\nuniform sampler2D texture;\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec4 tex = decodeHDR(texture2D(texture, v_Texcoord));\n\n gl_FragColor.rgb = tex.rgb;\n\n#ifdef OUTPUT_ALPHA\n gl_FragColor.a = tex.a;\n#else\n gl_FragColor.a = 1.0;\n#endif\n\n gl_FragColor = encodeHDR(gl_FragColor);\n\n #ifdef PREMULTIPLY_ALPHA\n gl_FragColor.rgb *= gl_FragColor.a;\n#endif\n}\n\n@end" }, function(e, t) { e.exports = "\n@export qtek.compositor.upsample\n\n#define HIGH_QUALITY\n\nuniform sampler2D texture;\nuniform vec2 textureSize : [512, 512];\n\nuniform float sampleScale: 0.5;\n\nvarying vec2 v_Texcoord;\n\n@import qtek.util.rgbm\n\n@import qtek.util.clamp_sample\n\nvoid main()\n{\n\n#ifdef HIGH_QUALITY\n vec4 d = vec4(1.0, 1.0, -1.0, 0.0) / textureSize.xyxy * sampleScale;\n\n vec4 s;\n s = decodeHDR(clampSample(texture, v_Texcoord - d.xy));\n s += decodeHDR(clampSample(texture, v_Texcoord - d.wy)) * 2.0;\n s += decodeHDR(clampSample(texture, v_Texcoord - d.zy));\n\n s += decodeHDR(clampSample(texture, v_Texcoord + d.zw)) * 2.0;\n s += decodeHDR(clampSample(texture, v_Texcoord )) * 4.0;\n s += decodeHDR(clampSample(texture, v_Texcoord + d.xw)) * 2.0;\n\n s += decodeHDR(clampSample(texture, v_Texcoord + d.zy));\n s += decodeHDR(clampSample(texture, v_Texcoord + d.wy)) * 2.0;\n s += decodeHDR(clampSample(texture, v_Texcoord + d.xy));\n\n gl_FragColor = encodeHDR(s / 16.0);\n#else\n vec4 d = vec4(-1.0, -1.0, +1.0, +1.0) / textureSize.xyxy;\n\n vec4 s;\n s = decodeHDR(clampSample(texture, v_Texcoord + d.xy));\n s += decodeHDR(clampSample(texture, v_Texcoord + d.zy));\n s += decodeHDR(clampSample(texture, v_Texcoord + d.xw));\n s += decodeHDR(clampSample(texture, v_Texcoord + d.zw));\n\n gl_FragColor = encodeHDR(s / 4.0);\n#endif\n}\n\n@end" }, function(e, t) { e.exports = "\n@export qtek.compositor.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position : POSITION;\nattribute vec2 texcoord : TEXCOORD_0;\n\nvarying vec2 v_Texcoord;\n\nvoid main()\n{\n v_Texcoord = texcoord;\n gl_Position = worldViewProjection * vec4(position, 1.0);\n}\n\n@end" }, function(e, t) { e.exports = "vec3 calcAmbientSHLight(int idx, vec3 N) {\n int offset = 9 * idx;\n\n return ambientSHLightCoefficients[0]\n + ambientSHLightCoefficients[1] * N.x\n + ambientSHLightCoefficients[2] * N.y\n + ambientSHLightCoefficients[3] * N.z\n + ambientSHLightCoefficients[4] * N.x * N.z\n + ambientSHLightCoefficients[5] * N.z * N.y\n + ambientSHLightCoefficients[6] * N.y * N.x\n + ambientSHLightCoefficients[7] * (3.0 * N.z * N.z - 1.0)\n + ambientSHLightCoefficients[8] * (N.x * N.x - N.y * N.y);\n}" }, function(e, t, r) {
        var n = ":unconfigurable;";
        e.exports = ["@export qtek.header.directional_light", "uniform vec3 directionalLightDirection[DIRECTIONAL_LIGHT_COUNT]" + n, "uniform vec3 directionalLightColor[DIRECTIONAL_LIGHT_COUNT]" + n, "@end", "@export qtek.header.ambient_light", "uniform vec3 ambientLightColor[AMBIENT_LIGHT_COUNT]" + n, "@end", "@export qtek.header.ambient_sh_light", "uniform vec3 ambientSHLightColor[AMBIENT_SH_LIGHT_COUNT]" + n, "uniform vec3 ambientSHLightCoefficients[AMBIENT_SH_LIGHT_COUNT * 9]" + n, r(223), "@end", "@export qtek.header.ambient_cubemap_light", "uniform vec3 ambientCubemapLightColor[AMBIENT_CUBEMAP_LIGHT_COUNT]" + n, "uniform samplerCube ambientCubemapLightCubemap[AMBIENT_CUBEMAP_LIGHT_COUNT]" + n, "uniform sampler2D ambientCubemapLightBRDFLookup[AMBIENT_CUBEMAP_LIGHT_COUNT]" + n, "@end", "@export qtek.header.point_light", "uniform vec3 pointLightPosition[POINT_LIGHT_COUNT]" + n, "uniform float pointLightRange[POINT_LIGHT_COUNT]" + n, "uniform vec3 pointLightColor[POINT_LIGHT_COUNT]" + n, "@end", "@export qtek.header.spot_light", "uniform vec3 spotLightPosition[SPOT_LIGHT_COUNT]" + n, "uniform vec3 spotLightDirection[SPOT_LIGHT_COUNT]" + n, "uniform float spotLightRange[SPOT_LIGHT_COUNT]" + n, "uniform float spotLightUmbraAngleCosine[SPOT_LIGHT_COUNT]" + n, "uniform float spotLightPenumbraAngleCosine[SPOT_LIGHT_COUNT]" + n, "uniform float spotLightFalloffFactor[SPOT_LIGHT_COUNT]" + n, "uniform vec3 spotLightColor[SPOT_LIGHT_COUNT]" + n, "@end"].join("\n")
    }, function(e, t) { e.exports = "@export qtek.sm.depth.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position : POSITION;\n\n#ifdef SHADOW_TRANSPARENT\nattribute vec2 texcoord : TEXCOORD_0;\n#endif\n\n@import qtek.chunk.skinning_header\n\nvarying vec4 v_ViewPosition;\n\n#ifdef SHADOW_TRANSPARENT\nvarying vec2 v_Texcoord;\n#endif\n\nvoid main(){\n\n vec3 skinnedPosition = position;\n\n#ifdef SKINNING\n\n @import qtek.chunk.skin_matrix\n\n skinnedPosition = (skinMatrixWS * vec4(position, 1.0)).xyz;\n#endif\n\n v_ViewPosition = worldViewProjection * vec4(skinnedPosition, 1.0);\n gl_Position = v_ViewPosition;\n\n#ifdef SHADOW_TRANSPARENT\n v_Texcoord = texcoord;\n#endif\n}\n@end\n\n@export qtek.sm.depth.fragment\n\nvarying vec4 v_ViewPosition;\n\n#ifdef SHADOW_TRANSPARENT\nvarying vec2 v_Texcoord;\n#endif\n\nuniform float bias : 0.001;\nuniform float slopeScale : 1.0;\n\n#ifdef SHADOW_TRANSPARENT\nuniform sampler2D transparentMap;\n#endif\n\n@import qtek.util.encode_float\n\nvoid main(){\n float depth = v_ViewPosition.z / v_ViewPosition.w;\n \n#ifdef USE_VSM\n depth = depth * 0.5 + 0.5;\n float moment1 = depth;\n float moment2 = depth * depth;\n\n float dx = dFdx(depth);\n float dy = dFdy(depth);\n moment2 += 0.25*(dx*dx+dy*dy);\n\n gl_FragColor = vec4(moment1, moment2, 0.0, 1.0);\n#else\n float dx = dFdx(depth);\n float dy = dFdy(depth);\n depth += sqrt(dx*dx + dy*dy) * slopeScale + bias;\n\n#ifdef SHADOW_TRANSPARENT\n if (texture2D(transparentMap, v_Texcoord).a <= 0.1) {\n gl_FragColor = encodeFloat(0.9999);\n return;\n }\n#endif\n\n gl_FragColor = encodeFloat(depth * 0.5 + 0.5);\n#endif\n}\n@end\n\n@export qtek.sm.debug_depth\n\nuniform sampler2D depthMap;\nvarying vec2 v_Texcoord;\n\n@import qtek.util.decode_float\n\nvoid main() {\n vec4 tex = texture2D(depthMap, v_Texcoord);\n#ifdef USE_VSM\n gl_FragColor = vec4(tex.rgb, 1.0);\n#else\n float depth = decodeFloat(tex);\n gl_FragColor = vec4(depth, depth, depth, 1.0);\n#endif\n}\n\n@end\n\n\n@export qtek.sm.distance.vertex\n\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\nuniform mat4 world : WORLD;\n\nattribute vec3 position : POSITION;\n\n@import qtek.chunk.skinning_header\n\nvarying vec3 v_WorldPosition;\n\nvoid main (){\n\n vec3 skinnedPosition = position;\n#ifdef SKINNING\n @import qtek.chunk.skin_matrix\n\n skinnedPosition = (skinMatrixWS * vec4(position, 1.0)).xyz;\n#endif\n\n gl_Position = worldViewProjection * vec4(skinnedPosition , 1.0);\n v_WorldPosition = (world * vec4(skinnedPosition, 1.0)).xyz;\n}\n\n@end\n\n@export qtek.sm.distance.fragment\n\nuniform vec3 lightPosition;\nuniform float range : 100;\n\nvarying vec3 v_WorldPosition;\n\n@import qtek.util.encode_float\n\nvoid main(){\n float dist = distance(lightPosition, v_WorldPosition);\n#ifdef USE_VSM\n gl_FragColor = vec4(dist, dist * dist, 0.0, 0.0);\n#else\n dist = dist / range;\n gl_FragColor = encodeFloat(dist);\n#endif\n}\n@end\n\n@export qtek.plugin.shadow_map_common\n\n@import qtek.util.decode_float\n\nfloat tapShadowMap(sampler2D map, vec2 uv, float z){\n vec4 tex = texture2D(map, uv);\n return step(z, decodeFloat(tex) * 2.0 - 1.0);\n}\n\nfloat pcf(sampler2D map, vec2 uv, float z, float textureSize, vec2 scale) {\n\n float shadowContrib = tapShadowMap(map, uv, z);\n vec2 offset = vec2(1.0 / textureSize) * scale;\n#ifdef PCF_KERNEL_SIZE\n for (int _idx_ = 0; _idx_ < PCF_KERNEL_SIZE; _idx_++) {{\n shadowContrib += tapShadowMap(map, uv + offset * pcfKernel[_idx_], z);\n }}\n\n return shadowContrib / float(PCF_KERNEL_SIZE + 1);\n#else\n shadowContrib += tapShadowMap(map, uv+vec2(offset.x, 0.0), z);\n shadowContrib += tapShadowMap(map, uv+vec2(offset.x, offset.y), z);\n shadowContrib += tapShadowMap(map, uv+vec2(-offset.x, offset.y), z);\n shadowContrib += tapShadowMap(map, uv+vec2(0.0, offset.y), z);\n shadowContrib += tapShadowMap(map, uv+vec2(-offset.x, 0.0), z);\n shadowContrib += tapShadowMap(map, uv+vec2(-offset.x, -offset.y), z);\n shadowContrib += tapShadowMap(map, uv+vec2(offset.x, -offset.y), z);\n shadowContrib += tapShadowMap(map, uv+vec2(0.0, -offset.y), z);\n\n return shadowContrib / 9.0;\n#endif\n}\n\nfloat pcf(sampler2D map, vec2 uv, float z, float textureSize) {\n return pcf(map, uv, z, textureSize, vec2(1.0));\n}\n\nfloat chebyshevUpperBound(vec2 moments, float z){\n float p = 0.0;\n z = z * 0.5 + 0.5;\n if (z <= moments.x) {\n p = 1.0;\n }\n float variance = moments.y - moments.x * moments.x;\n variance = max(variance, 0.0000001);\n float mD = moments.x - z;\n float pMax = variance / (variance + mD * mD);\n pMax = clamp((pMax-0.4)/(1.0-0.4), 0.0, 1.0);\n return max(p, pMax);\n}\nfloat computeShadowContrib(\n sampler2D map, mat4 lightVPM, vec3 position, float textureSize, vec2 scale, vec2 offset\n) {\n\n vec4 posInLightSpace = lightVPM * vec4(position, 1.0);\n posInLightSpace.xyz /= posInLightSpace.w;\n float z = posInLightSpace.z;\n if(all(greaterThan(posInLightSpace.xyz, vec3(-0.99, -0.99, -1.0))) &&\n all(lessThan(posInLightSpace.xyz, vec3(0.99, 0.99, 1.0)))){\n vec2 uv = (posInLightSpace.xy+1.0) / 2.0;\n\n #ifdef USE_VSM\n vec2 moments = texture2D(map, uv * scale + offset).xy;\n return chebyshevUpperBound(moments, z);\n #else\n return pcf(map, uv * scale + offset, z, textureSize, scale);\n #endif\n }\n return 1.0;\n}\n\nfloat computeShadowContrib(sampler2D map, mat4 lightVPM, vec3 position, float textureSize) {\n return computeShadowContrib(map, lightVPM, position, textureSize, vec2(1.0), vec2(0.0));\n}\n\nfloat computeShadowContribOmni(samplerCube map, vec3 direction, float range)\n{\n float dist = length(direction);\n vec4 shadowTex = textureCube(map, direction);\n\n#ifdef USE_VSM\n vec2 moments = shadowTex.xy;\n float variance = moments.y - moments.x * moments.x;\n float mD = moments.x - dist;\n float p = variance / (variance + mD * mD);\n if(moments.x + 0.001 < dist){\n return clamp(p, 0.0, 1.0);\n }else{\n return 1.0;\n }\n#else\n return step(dist, (decodeFloat(shadowTex) + 0.0002) * range);\n#endif\n}\n\n@end\n\n\n\n@export qtek.plugin.compute_shadow_map\n\n#if defined(SPOT_LIGHT_SHADOWMAP_COUNT) || defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT) || defined(POINT_LIGHT_SHADOWMAP_COUNT)\n\n#ifdef SPOT_LIGHT_SHADOWMAP_COUNT\nuniform sampler2D spotLightShadowMaps[SPOT_LIGHT_SHADOWMAP_COUNT];\nuniform mat4 spotLightMatrices[SPOT_LIGHT_SHADOWMAP_COUNT];\nuniform float spotLightShadowMapSizes[SPOT_LIGHT_SHADOWMAP_COUNT];\n#endif\n\n#ifdef DIRECTIONAL_LIGHT_SHADOWMAP_COUNT\n#if defined(SHADOW_CASCADE)\nuniform sampler2D directionalLightShadowMaps[1];\nuniform mat4 directionalLightMatrices[SHADOW_CASCADE];\nuniform float directionalLightShadowMapSizes[1];\nuniform float shadowCascadeClipsNear[SHADOW_CASCADE];\nuniform float shadowCascadeClipsFar[SHADOW_CASCADE];\n#else\nuniform sampler2D directionalLightShadowMaps[DIRECTIONAL_LIGHT_SHADOWMAP_COUNT];\nuniform mat4 directionalLightMatrices[DIRECTIONAL_LIGHT_SHADOWMAP_COUNT];\nuniform float directionalLightShadowMapSizes[DIRECTIONAL_LIGHT_SHADOWMAP_COUNT];\n#endif\n#endif\n\n#ifdef POINT_LIGHT_SHADOWMAP_COUNT\nuniform samplerCube pointLightShadowMaps[POINT_LIGHT_SHADOWMAP_COUNT];\nuniform float pointLightShadowMapSizes[POINT_LIGHT_SHADOWMAP_COUNT];\n#endif\n\nuniform bool shadowEnabled : true;\n\n#ifdef PCF_KERNEL_SIZE\nuniform vec2 pcfKernel[PCF_KERNEL_SIZE];\n#endif\n\n@import qtek.plugin.shadow_map_common\n\n#if defined(SPOT_LIGHT_SHADOWMAP_COUNT)\n\nvoid computeShadowOfSpotLights(vec3 position, inout float shadowContribs[SPOT_LIGHT_COUNT] ) {\n float shadowContrib;\n for(int _idx_ = 0; _idx_ < SPOT_LIGHT_SHADOWMAP_COUNT; _idx_++) {{\n shadowContrib = computeShadowContrib(\n spotLightShadowMaps[_idx_], spotLightMatrices[_idx_], position,\n spotLightShadowMapSizes[_idx_]\n );\n shadowContribs[_idx_] = shadowContrib;\n }}\n for(int _idx_ = SPOT_LIGHT_SHADOWMAP_COUNT; _idx_ < SPOT_LIGHT_COUNT; _idx_++){{\n shadowContribs[_idx_] = 1.0;\n }}\n}\n\n#endif\n\n\n#if defined(DIRECTIONAL_LIGHT_SHADOWMAP_COUNT)\n\n#ifdef SHADOW_CASCADE\n\nvoid computeShadowOfDirectionalLights(vec3 position, inout float shadowContribs[DIRECTIONAL_LIGHT_COUNT]){\n float depth = (2.0 * gl_FragCoord.z - gl_DepthRange.near - gl_DepthRange.far)\n / (gl_DepthRange.far - gl_DepthRange.near);\n\n float shadowContrib;\n shadowContribs[0] = 1.0;\n\n for (int _idx_ = 0; _idx_ < SHADOW_CASCADE; _idx_++) {{\n if (\n depth >= shadowCascadeClipsNear[_idx_] &&\n depth <= shadowCascadeClipsFar[_idx_]\n ) {\n shadowContrib = computeShadowContrib(\n directionalLightShadowMaps[0], directionalLightMatrices[_idx_], position,\n directionalLightShadowMapSizes[0],\n vec2(1.0 / float(SHADOW_CASCADE), 1.0),\n vec2(float(_idx_) / float(SHADOW_CASCADE), 0.0)\n );\n shadowContribs[0] = shadowContrib;\n }\n }}\n for(int _idx_ = DIRECTIONAL_LIGHT_SHADOWMAP_COUNT; _idx_ < DIRECTIONAL_LIGHT_COUNT; _idx_++) {{\n shadowContribs[_idx_] = 1.0;\n }}\n}\n\n#else\n\nvoid computeShadowOfDirectionalLights(vec3 position, inout float shadowContribs[DIRECTIONAL_LIGHT_COUNT]){\n float shadowContrib;\n\n for(int _idx_ = 0; _idx_ < DIRECTIONAL_LIGHT_SHADOWMAP_COUNT; _idx_++) {{\n shadowContrib = computeShadowContrib(\n directionalLightShadowMaps[_idx_], directionalLightMatrices[_idx_], position,\n directionalLightShadowMapSizes[_idx_]\n );\n shadowContribs[_idx_] = shadowContrib;\n }}\n for(int _idx_ = DIRECTIONAL_LIGHT_SHADOWMAP_COUNT; _idx_ < DIRECTIONAL_LIGHT_COUNT; _idx_++) {{\n shadowContribs[_idx_] = 1.0;\n }}\n}\n#endif\n\n#endif\n\n\n#if defined(POINT_LIGHT_SHADOWMAP_COUNT)\n\nvoid computeShadowOfPointLights(vec3 position, inout float shadowContribs[POINT_LIGHT_COUNT] ){\n vec3 lightPosition;\n vec3 direction;\n for(int _idx_ = 0; _idx_ < POINT_LIGHT_SHADOWMAP_COUNT; _idx_++) {{\n lightPosition = pointLightPosition[_idx_];\n direction = position - lightPosition;\n shadowContribs[_idx_] = computeShadowContribOmni(pointLightShadowMaps[_idx_], direction, pointLightRange[_idx_]);\n }}\n for(int _idx_ = POINT_LIGHT_SHADOWMAP_COUNT; _idx_ < POINT_LIGHT_COUNT; _idx_++) {{\n shadowContribs[_idx_] = 1.0;\n }}\n}\n\n#endif\n\n#endif\n\n@end" }, function(e, t) { e.exports = "@export qtek.skybox.vertex\n\nuniform mat4 world : WORLD;\nuniform mat4 worldViewProjection : WORLDVIEWPROJECTION;\n\nattribute vec3 position : POSITION;\n\nvarying vec3 v_WorldPosition;\n\nvoid main()\n{\n v_WorldPosition = (world * vec4(position, 1.0)).xyz;\n gl_Position = worldViewProjection * vec4(position, 1.0);\n}\n\n@end\n\n@export qtek.skybox.fragment\n\nuniform mat4 viewInverse : VIEWINVERSE;\nuniform samplerCube environmentMap;\nuniform float lod: 0.0;\n\nvarying vec3 v_WorldPosition;\n\n@import qtek.util.rgbm\n\nvoid main()\n{\n vec3 eyePos = viewInverse[3].xyz;\n vec3 viewDirection = normalize(v_WorldPosition - eyePos);\n\n vec3 tex = decodeHDR(textureCubeLodEXT(environmentMap, viewDirection, lod)).rgb;\n\n#ifdef SRGB_DECODE\n tex.rgb = pow(tex.rgb, vec3(2.2));\n#endif\n\n gl_FragColor = encodeHDR(vec4(tex, 1.0));\n}\n@end" }, function(e, t) { e.exports = "\n@export qtek.util.rand\nhighp float rand(vec2 uv) {\n const highp float a = 12.9898, b = 78.233, c = 43758.5453;\n highp float dt = dot(uv.xy, vec2(a,b)), sn = mod(dt, 3.141592653589793);\n return fract(sin(sn) * c);\n}\n@end\n\n@export qtek.util.calculate_attenuation\n\nuniform float attenuationFactor : 5.0;\n\nfloat lightAttenuation(float dist, float range)\n{\n float attenuation = 1.0;\n attenuation = dist*dist/(range*range+1.0);\n float att_s = attenuationFactor;\n attenuation = 1.0/(attenuation*att_s+1.0);\n att_s = 1.0/(att_s+1.0);\n attenuation = attenuation - att_s;\n attenuation /= 1.0 - att_s;\n return clamp(attenuation, 0.0, 1.0);\n}\n\n@end\n\n@export qtek.util.edge_factor\n\nfloat edgeFactor(float width)\n{\n vec3 d = fwidth(v_Barycentric);\n vec3 a3 = smoothstep(vec3(0.0), d * width, v_Barycentric);\n return min(min(a3.x, a3.y), a3.z);\n}\n\n@end\n\n@export qtek.util.encode_float\nvec4 encodeFloat(const in float depth)\n{\n \n \n const vec4 bitShifts = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);\n const vec4 bit_mask = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);\n vec4 res = fract(depth * bitShifts);\n res -= res.xxyz * bit_mask;\n\n return res;\n}\n@end\n\n@export qtek.util.decode_float\nfloat decodeFloat(const in vec4 color)\n{\n \n \n const vec4 bitShifts = vec4(1.0/(256.0*256.0*256.0), 1.0/(256.0*256.0), 1.0/256.0, 1.0);\n return dot(color, bitShifts);\n}\n@end\n\n\n@export qtek.util.float\n@import qtek.util.encode_float\n@import qtek.util.decode_float\n@end\n\n\n\n@export qtek.util.rgbm_decode\nvec3 RGBMDecode(vec4 rgbm, float range) {\n return range * rgbm.rgb * rgbm.a;\n }\n@end\n\n@export qtek.util.rgbm_encode\nvec4 RGBMEncode(vec3 color, float range) {\n if (dot(color, color) == 0.0) {\n return vec4(0.0);\n }\n vec4 rgbm;\n color /= range;\n rgbm.a = clamp(max(max(color.r, color.g), max(color.b, 1e-6)), 0.0, 1.0);\n rgbm.a = ceil(rgbm.a * 255.0) / 255.0;\n rgbm.rgb = color / rgbm.a;\n return rgbm;\n}\n@end\n\n@export qtek.util.rgbm\n@import qtek.util.rgbm_decode\n@import qtek.util.rgbm_encode\n\nvec4 decodeHDR(vec4 color)\n{\n#if defined(RGBM_DECODE) || defined(RGBM)\n return vec4(RGBMDecode(color, 51.5), 1.0);\n#else\n return color;\n#endif\n}\n\nvec4 encodeHDR(vec4 color)\n{\n#if defined(RGBM_ENCODE) || defined(RGBM)\n return RGBMEncode(color.xyz, 51.5);\n#else\n return color;\n#endif\n}\n\n@end\n\n\n@export qtek.util.srgb\n\nvec4 sRGBToLinear(in vec4 value) {\n return vec4(mix(pow(value.rgb * 0.9478672986 + vec3(0.0521327014), vec3(2.4)), value.rgb * 0.0773993808, vec3(lessThanEqual(value.rgb, vec3(0.04045)))), value.w);\n}\n\nvec4 linearTosRGB(in vec4 value) {\n return vec4(mix(pow(value.rgb, vec3(0.41666)) * 1.055 - vec3(0.055), value.rgb * 12.92, vec3(lessThanEqual(value.rgb, vec3(0.0031308)))), value.w);\n}\n@end\n\n\n@export qtek.chunk.skinning_header\n#ifdef SKINNING\nattribute vec3 weight : WEIGHT;\nattribute vec4 joint : JOINT;\n\n#ifdef USE_SKIN_MATRICES_TEXTURE\nuniform sampler2D skinMatricesTexture;\nuniform float skinMatricesTextureSize: unconfigurable;\nmat4 getSkinMatrix(float idx) {\n float j = idx * 4.0;\n float x = mod(j, skinMatricesTextureSize);\n float y = floor(j / skinMatricesTextureSize) + 0.5;\n vec2 scale = vec2(skinMatricesTextureSize);\n\n return mat4(\n texture2D(skinMatricesTexture, vec2(x + 0.5, y) / scale),\n texture2D(skinMatricesTexture, vec2(x + 1.5, y) / scale),\n texture2D(skinMatricesTexture, vec2(x + 2.5, y) / scale),\n texture2D(skinMatricesTexture, vec2(x + 3.5, y) / scale)\n );\n}\n#else\nuniform mat4 skinMatrix[JOINT_COUNT] : SKIN_MATRIX;\nmat4 getSkinMatrix(float idx) {\n return skinMatrix[int(idx)];\n}\n#endif\n\n#endif\n\n@end\n\n@export qtek.chunk.skin_matrix\n\nmat4 skinMatrixWS;\nif (weight.x > 1e-4)\n{\n skinMatrixWS = getSkinMatrix(joint.x) * weight.x;\n}\nif (weight.y > 1e-4)\n{\n skinMatrixWS += getSkinMatrix(joint.y) * weight.y;\n}\nif (weight.z > 1e-4)\n{\n skinMatrixWS += getSkinMatrix(joint.z) * weight.z;\n}\nfloat weightW = 1.0-weight.x-weight.y-weight.z;\nif (weightW > 1e-3)\n{\n skinMatrixWS += getSkinMatrix(joint.w) * weightW;\n}\n@end\n\n\n\n@export qtek.util.parallax_correct\n\nvec3 parallaxCorrect(in vec3 dir, in vec3 pos, in vec3 boxMin, in vec3 boxMax) {\n vec3 first = (boxMax - pos) / dir;\n vec3 second = (boxMin - pos) / dir;\n\n vec3 further = max(first, second);\n float dist = min(further.x, min(further.y, further.z));\n\n vec3 fixedPos = pos + dir * dist;\n vec3 boxCenter = (boxMax + boxMin) * 0.5;\n\n return normalize(fixedPos - boxCenter);\n}\n\n@end\n\n\n\n@export qtek.util.clamp_sample\nvec4 clampSample(const in sampler2D texture, const in vec2 coord)\n{\n#ifdef STEREO\n float eye = step(0.5, coord.x) * 0.5;\n vec2 coordClamped = clamp(coord, vec2(eye, 0.0), vec2(0.5 + eye, 1.0));\n#else\n vec2 coordClamped = clamp(coord, vec2(0.0), vec2(1.0));\n#endif\n return texture2D(texture, coordClamped);\n}\n@end" }, function(e, t, r) {
        var n = r(5),
            i = r(23),
            a = r(6),
            o = r(10),
            s = r(12),
            u = r(16),
            h = r(7),
            l = r(57),
            c = r(26),
            d = r(59),
            f = r(20),
            p = r(47),
            _ = r(232),
            m = r(233),
            g = {},
            v = ["px", "nx", "py", "ny", "pz", "nz"];
        g.prefilterEnvironmentMap = function(e, t, r, s, _) {
            _ && s || (s = g.generateNormalDistribution(), _ = g.integrateBRDF(e, s)), r = r || {};
            var y = r.width || 64,
                x = r.height || 64,
                T = r.type || t.type,
                b = new i({ width: y, height: x, type: T, flipY: !1, mipmaps: [] });
            b.isPowerOfTwo() || console.warn("Width and height must be power of two to enable mipmap.");
            var w = Math.min(y, x),
                E = Math.log(w) / Math.log(2) + 1,
                S = new u({ shader: new h({ vertex: h.source("qtek.skybox.vertex"), fragment: m }) });
            S.set("normalDistribution", s), r.encodeRGBM && S.shader.define("fragment", "RGBM_ENCODE"), r.decodeRGBM && S.shader.define("fragment", "RGBM_DECODE");
            var A, M = new c;
            if (t instanceof n) {
                var N = new i({ width: y, height: x, type: T === a.FLOAT ? a.HALF_FLOAT : T });
                p.panoramaToCubeMap(e, t, N, { encodeRGBM: r.decodeRGBM }), t = N
            }
            A = new l({ scene: M, material: S }), A.material.set("environmentMap", t);
            var C = new d({ texture: b });
            r.encodeRGBM && (T = b.type = a.UNSIGNED_BYTE);
            for (var L = new n({ width: y, height: x, type: T }), D = new o({ depthBuffer: !1 }), I = f[T === a.UNSIGNED_BYTE ? "Uint8Array" : "Float32Array"], R = 0; R < E; R++) {
                b.mipmaps[R] = { pixels: {} }, A.material.set("roughness", R / (v.length - 1));
                for (var P = L.width, O = 2 * Math.atan(P / (P - .5)) / Math.PI * 180, F = 0; F < v.length; F++) {
                    var B = new I(L.width * L.height * 4);
                    D.attach(L), D.bind(e);
                    var U = C.getCamera(v[F]);
                    U.fov = O, e.render(M, U), e.gl.readPixels(0, 0, L.width, L.height, a.RGBA, T, B), D.unbind(e), b.mipmaps[R].pixels[v[F]] = B
                }
                L.width /= 2, L.height /= 2, L.dirty()
            }
            return D.dispose(e.gl), L.dispose(e.gl), A.dispose(e.gl), s.dispose(e.gl), { environmentMap: b, brdfLookup: _, normalDistribution: s, maxMipmapLevel: E }
        }, g.integrateBRDF = function(e, t) {
            t = t || g.generateNormalDistribution();
            var r = new o({ depthBuffer: !1 }),
                i = new s({ fragment: _ }),
                u = new n({ width: 512, height: 256, type: a.HALF_FLOAT, minFilter: a.NEAREST, magFilter: a.NEAREST, useMipmap: !1 });
            return i.setUniform("normalDistribution", t), i.setUniform("viewportSize", [512, 256]), i.attachOutput(u), i.render(e, r), r.dispose(e.gl), u
        }, g.generateNormalDistribution = function(e, t) {
            for (var e = e || 256, t = t || 1024, r = new n({ width: e, height: t, type: a.FLOAT, minFilter: a.NEAREST, magFilter: a.NEAREST, useMipmap: !1 }), i = new Float32Array(t * e * 4), o = 0; o < t; o++) {
                var s = o / t,
                    u = (o << 16 | o >>> 16) >>> 0;
                u = ((1431655765 & u) << 1 | (2863311530 & u) >>> 1) >>> 0, u = ((858993459 & u) << 2 | (3435973836 & u) >>> 2) >>> 0, u = ((252645135 & u) << 4 | (4042322160 & u) >>> 4) >>> 0, u = (((16711935 & u) << 8 | (4278255360 & u) >>> 8) >>> 0) / 4294967296;
                for (var h = 0; h < e; h++) {
                    var l = h / e,
                        c = l * l,
                        d = 2 * Math.PI * s,
                        f = Math.sqrt((1 - u) / (1 + (c * c - 1) * u)),
                        p = Math.sqrt(1 - f * f),
                        _ = 4 * (o * e + h);
                    i[_] = p * Math.cos(d), i[_ + 1] = p * Math.sin(d), i[_ + 2] = f, i[_ + 3] = 1
                }
            }
            return r.pixels = i, r
        }, e.exports = g
    }, function(e, t, r) {
        "use strict";

        function n(e) { return e.charCodeAt(0) + (e.charCodeAt(1) << 8) + (e.charCodeAt(2) << 16) + (e.charCodeAt(3) << 24) }
        var i = r(6),
            a = r(5),
            o = (r(23), n("DXT1")),
            s = n("DXT3"),
            u = n("DXT5"),
            h = {
                parse: function(e, t) {
                    var r = new Int32Array(e, 0, 31);
                    if (542327876 !== r[0]) return null;
                    if (4 & !r(20)) return null;
                    var n, h, l = r(21),
                        c = r[4],
                        d = r[3],
                        f = 512 & r[28],
                        p = 131072 & r[2];
                    switch (l) {
                        case o:
                            n = 8, h = i.COMPRESSED_RGB_S3TC_DXT1_EXT;
                            break;
                        case s:
                            n = 16, h = i.COMPRESSED_RGBA_S3TC_DXT3_EXT;
                            break;
                        case u:
                            n = 16, h = i.COMPRESSED_RGBA_S3TC_DXT5_EXT;
                            break;
                        default:
                            return null
                    }
                    var _ = r[1] + 4,
                        m = f ? 6 : 1,
                        g = 1;
                    p && (g = Math.max(1, r[7]));
                    for (var v = [], y = 0; y < m; y++) {
                        var x = c,
                            T = d;
                        v[y] = new a({ width: x, height: T, format: h });
                        for (var b = [], w = 0; w < g; w++) {
                            var E = Math.max(4, x) / 4 * Math.max(4, T) / 4 * n,
                                S = new Uint8Array(e, _, E);
                            _ += E, x *= .5, T *= .5, b[w] = S
                        }
                        v[y].pixels = b[0], p && (v[y].mipmaps = b)
                    }
                    if (!t) return v[0];
                    t.width = v[0].width, t.height = v[0].height, t.format = v[0].format, t.pixels = v[0].pixels, t.mipmaps = v[0].mipmaps
                }
            };
        e.exports = h
    }, function(e, t, r) {
        "use strict";

        function n(e, t, r, n) {
            if (e[3] > 0) {
                var i = Math.pow(2, e[3] - 128 - 8 + n);
                t[r + 0] = e[0] * i, t[r + 1] = e[1] * i, t[r + 2] = e[2] * i
            } else t[r + 0] = 0, t[r + 1] = 0, t[r + 2] = 0;
            return t[r + 3] = 1, t
        }

        function i(e, t, r) { for (var n = "", i = t; i < r; i++) n += l(e[i]); return n }

        function a(e, t) { t[0] = e[0], t[1] = e[1], t[2] = e[2], t[3] = e[3] }

        function o(e, t, r, n) {
            for (var i = 0, o = 0, s = n; s > 0;)
                if (e[o][0] = t[r++], e[o][1] = t[r++], e[o][2] = t[r++], e[o][3] = t[r++], 1 === e[o][0] && 1 === e[o][1] && 1 === e[o][2]) {
                    for (var u = e[o][3] << i >>> 0; u > 0; u--) a(e[o - 1], e[o]), o++, s--;
                    i += 8
                } else o++, s--, i = 0;
            return r
        }

        function s(e, t, r, n) {
            if (n < c | n > d) return o(e, t, r, n);
            var i = t[r++];
            if (2 != i) return o(e, t, r - 1, n);
            if (e[0][1] = t[r++], e[0][2] = t[r++], i = t[r++], (e[0][2] << 8 >>> 0 | i) >>> 0 !== n) return null;
            for (var i = 0; i < 4; i++)
                for (var a = 0; a < n;) {
                    var s = t[r++];
                    if (s > 128) { s = (127 & s) >>> 0; for (var u = t[r++]; s--;) e[a++][i] = u } else
                        for (; s--;) e[a++][i] = t[r++]
                }
            return r
        }
        var u = r(6),
            h = r(5),
            l = String.fromCharCode,
            c = 8,
            d = 32767,
            f = {
                parseRGBE: function(e, t, r) {
                    null == r && (r = 0);
                    var a = new Uint8Array(e),
                        o = a.length;
                    if ("#?" === i(a, 0, 2)) {
                        for (var c = 2; c < o && ("\n" !== l(a[c]) || "\n" !== l(a[c + 1])); c++);
                        if (!(c >= o)) {
                            c += 2;
                            for (var d = ""; c < o; c++) {
                                var f = l(a[c]);
                                if ("\n" === f) break;
                                d += f
                            }
                            var p = d.split(" "),
                                _ = parseInt(p[1]),
                                m = parseInt(p[3]);
                            if (m && _) { for (var g = c + 1, v = [], y = 0; y < m; y++) { v[y] = []; for (var x = 0; x < 4; x++) v[y][x] = 0 } for (var T = new Float32Array(m * _ * 4), b = 0, w = 0; w < _; w++) { var g = s(v, a, g, m); if (!g) return null; for (var y = 0; y < m; y++) n(v[y], T, b, r), b += 4 } return t || (t = new h), t.width = m, t.height = _, t.pixels = T, t.type = u.FLOAT, t }
                        }
                    }
                },
                parseRGBEFromPNG: function(e) {}
            };
        e.exports = f
    }, function(e, t, r) {
        function n(e, t) {
            var r = e[0],
                n = e[1],
                i = e[2];
            return 0 === t ? 1 : 1 === t ? r : 2 === t ? n : 3 === t ? i : 4 === t ? r * i : 5 === t ? n * i : 6 === t ? r * n : 7 === t ? 3 * i * i - 1 : r * r - n * n
        }

        function i(e, t, r, i) {
            for (var a = new u.Float32Array(27), o = p.create(), s = p.create(), h = p.create(), l = 0; l < 9; l++) {
                for (var c = p.create(), d = 0; d < m.length; d++) {
                    for (var f = t[m[d]], _ = p.create(), v = 0, y = 0, x = g[m[d]], T = 0; T < i; T++)
                        for (var b = 0; b < r; b++) {
                            o[0] = b / (r - 1) * 2 - 1, o[1] = T / (i - 1) * 2 - 1, o[2] = -1, p.normalize(o, o), h[0] = o[x[0]] * x[3], h[1] = o[x[1]] * x[4], h[2] = o[x[2]] * x[5], s[0] = f[y++] / 255, s[1] = f[y++] / 255, s[2] = f[y++] / 255;
                            var w = f[y++] / 255 * 51.5;
                            s[0] *= w, s[1] *= w, s[2] *= w, p.scaleAndAdd(_, _, s, n(h, l) * -o[2]), v += -o[2]
                        }
                    p.scaleAndAdd(c, c, _, 1 / v)
                }
                a[3 * l] = c[0] / 6, a[3 * l + 1] = c[1] / 6, a[3 * l + 2] = c[2] / 6
            }
            return a
        }
        var a = r(6),
            o = r(10),
            s = r(5),
            u = (r(23), r(47), r(12), r(20)),
            h = r(57),
            l = r(58),
            c = r(59),
            d = r(26),
            f = r(1),
            p = f.vec3,
            _ = {},
            m = (r(234), ["px", "nx", "py", "ny", "pz", "nz"]),
            g = { px: [2, 1, 0, -1, -1, 1], nx: [2, 1, 0, 1, -1, -1], py: [0, 2, 1, 1, -1, -1], ny: [0, 2, 1, 1, 1, 1], pz: [0, 1, 2, -1, -1, -1], nz: [0, 1, 2, 1, -1, 1] };
        _.projectEnvironmentMap = function(e, t, r) {
            r = r || {}, r.lod = r.lod || 0;
            var n, u = new d,
                f = 64;
            t instanceof s ? n = new l({ scene: u, environmentMap: t }) : (f = t.image && t.image.px ? t.image.px.width : t.width, n = new h({ scene: u, environmentMap: t }));
            var p = Math.ceil(f / Math.pow(2, r.lod)),
                _ = Math.ceil(f / Math.pow(2, r.lod)),
                g = new s({ width: p, height: _ }),
                v = new o;
            n.material.shader.define("fragment", "RGBM_ENCODE"), r.decodeRGBM && n.material.shader.define("fragment", "RGBM_DECODE"), n.material.set("lod", r.lod);
            for (var y = new c({ texture: g }), x = {}, T = 0; T < m.length; T++) {
                x[m[T]] = new Uint8Array(p * _ * 4);
                var b = y.getCamera(m[T]);
                b.fov = 90, v.attach(g), v.bind(e), e.render(u, b), e.gl.readPixels(0, 0, p, _, a.RGBA, a.UNSIGNED_BYTE, x[m[T]]), v.unbind(e)
            }
            return n.dispose(e.gl), v.dispose(e.gl), g.dispose(e.gl), i(e, x, p, _)
        }, e.exports = _
    }, function(e, t) { e.exports = "#define SAMPLE_NUMBER 1024\n#define PI 3.14159265358979\n\n\nuniform sampler2D normalDistribution;\n\nuniform vec2 viewportSize : [512, 256];\n\nconst vec3 N = vec3(0.0, 0.0, 1.0);\nconst float fSampleNumber = float(SAMPLE_NUMBER);\n\nvec3 importanceSampleNormal(float i, float roughness, vec3 N) {\n vec3 H = texture2D(normalDistribution, vec2(roughness, i)).rgb;\n\n vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n vec3 tangentX = normalize(cross(upVector, N));\n vec3 tangentY = cross(N, tangentX);\n return tangentX * H.x + tangentY * H.y + N * H.z;\n}\n\nfloat G_Smith(float roughness, float NoV, float NoL) {\n float k = roughness * roughness / 2.0;\n float G1V = NoV / (NoV * (1.0 - k) + k);\n float G1L = NoL / (NoL * (1.0 - k) + k);\n return G1L * G1V;\n}\n\nvoid main() {\n vec2 uv = gl_FragCoord.xy / viewportSize;\n float NoV = uv.x;\n float roughness = uv.y;\n\n vec3 V;\n V.x = sqrt(1.0 - NoV * NoV);\n V.y = 0.0;\n V.z = NoV;\n\n float A = 0.0;\n float B = 0.0;\n\n for (int i = 0; i < SAMPLE_NUMBER; i++) {\n vec3 H = importanceSampleNormal(float(i) / fSampleNumber, roughness, N);\n vec3 L = reflect(-V, H);\n float NoL = clamp(L.z, 0.0, 1.0);\n float NoH = clamp(H.z, 0.0, 1.0);\n float VoH = clamp(dot(V, H), 0.0, 1.0);\n\n if (NoL > 0.0) {\n float G = G_Smith(roughness, NoV, NoL);\n float G_Vis = G * VoH / (NoH * NoV);\n float Fc = pow(1.0 - VoH, 5.0);\n A += (1.0 - Fc) * G_Vis;\n B += Fc * G_Vis;\n }\n }\n\n gl_FragColor = vec4(vec2(A, B) / fSampleNumber, 0.0, 1.0);\n}\n" }, function(e, t) { e.exports = "#define SAMPLE_NUMBER 1024\n#define PI 3.14159265358979\n\nuniform mat4 viewInverse : VIEWINVERSE;\nuniform samplerCube environmentMap;\nuniform sampler2D normalDistribution;\n\nuniform float roughness : 0.5;\n\nvarying vec2 v_Texcoord;\nvarying vec3 v_WorldPosition;\n\nconst float fSampleNumber = float(SAMPLE_NUMBER);\n\n@import qtek.util.rgbm\n\nvec3 importanceSampleNormal(float i, float roughness, vec3 N) {\n vec3 H = texture2D(normalDistribution, vec2(roughness, i)).rgb;\n\n vec3 upVector = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);\n vec3 tangentX = normalize(cross(upVector, N));\n vec3 tangentY = cross(N, tangentX);\n return tangentX * H.x + tangentY * H.y + N * H.z;\n}\n\nvoid main() {\n\n vec3 eyePos = viewInverse[3].xyz;\n vec3 V = normalize(v_WorldPosition - eyePos);\n\n vec3 N = V;\n vec3 R = V;\n\n vec3 prefilteredColor = vec3(0.0);\n float totalWeight = 0.0;\n\n\n for (int i = 0; i < SAMPLE_NUMBER; i++) {\n vec3 H = importanceSampleNormal(float(i) / fSampleNumber, roughness, N);\n vec3 L = reflect(-V, H);\n\n float NoL = clamp(dot(N, L), 0.0, 1.0);\n if (NoL > 0.0) {\n prefilteredColor += decodeHDR(textureCube(environmentMap, L)).rgb * NoL;\n totalWeight += NoL;\n }\n }\n\n gl_FragColor = encodeHDR(vec4(prefilteredColor / totalWeight, 1.0));\n}\n" }, function(e, t) { e.exports = "uniform samplerCube environmentMap;\n\nvarying vec2 v_Texcoord;\n\n#define TEXTURE_SIZE 16\n\nmat3 front = mat3(\n 1.0, 0.0, 0.0,\n 0.0, 1.0, 0.0,\n 0.0, 0.0, 1.0\n);\n\nmat3 back = mat3(\n -1.0, 0.0, 0.0,\n 0.0, 1.0, 0.0,\n 0.0, 0.0, -1.0\n);\n\nmat3 left = mat3(\n 0.0, 0.0, -1.0,\n 0.0, 1.0, 0.0,\n 1.0, 0.0, 0.0\n);\n\nmat3 right = mat3(\n 0.0, 0.0, 1.0,\n 0.0, 1.0, 0.0,\n -1.0, 0.0, 0.0\n);\n\nmat3 up = mat3(\n 1.0, 0.0, 0.0,\n 0.0, 0.0, 1.0,\n 0.0, -1.0, 0.0\n);\n\nmat3 down = mat3(\n 1.0, 0.0, 0.0,\n 0.0, 0.0, -1.0,\n 0.0, 1.0, 0.0\n);\n\n\nfloat harmonics(vec3 normal){\n int index = int(gl_FragCoord.x);\n\n float x = normal.x;\n float y = normal.y;\n float z = normal.z;\n\n if(index==0){\n return 1.0;\n }\n else if(index==1){\n return x;\n }\n else if(index==2){\n return y;\n }\n else if(index==3){\n return z;\n }\n else if(index==4){\n return x*z;\n }\n else if(index==5){\n return y*z;\n }\n else if(index==6){\n return x*y;\n }\n else if(index==7){\n return 3.0*z*z - 1.0;\n }\n else{\n return x*x - y*y;\n }\n}\n\nvec3 sampleSide(mat3 rot)\n{\n\n vec3 result = vec3(0.0);\n float divider = 0.0;\n for (int i = 0; i < TEXTURE_SIZE * TEXTURE_SIZE; i++) {\n float x = mod(float(i), float(TEXTURE_SIZE));\n float y = float(i / TEXTURE_SIZE);\n\n vec2 sidecoord = ((vec2(x, y) + vec2(0.5, 0.5)) / vec2(TEXTURE_SIZE)) * 2.0 - 1.0;\n vec3 normal = normalize(vec3(sidecoord, -1.0));\n vec3 fetchNormal = rot * normal;\n vec3 texel = textureCube(environmentMap, fetchNormal).rgb;\n\n result += harmonics(fetchNormal) * texel * -normal.z;\n\n divider += -normal.z;\n }\n\n return result / divider;\n}\n\nvoid main()\n{\n vec3 result = (\n sampleSide(front) +\n sampleSide(back) +\n sampleSide(left) +\n sampleSide(right) +\n sampleSide(up) +\n sampleSide(down)\n ) / 6.0;\n gl_FragColor = vec4(result, 1.0);\n}" }, function(e, t) { e.exports = "0.4.3" }, function(e, t, r) {
        function n(e, t) { return e[t] }

        function i(e, t, r) { e[t] = r }

        function a(e, t, r) { return (t - e) * r + e }

        function o(e, t, r) { return r > .5 ? t : e }

        function s(e, t, r, n, i) {
            var o = e.length;
            if (1 == i)
                for (var s = 0; s < o; s++) n[s] = a(e[s], t[s], r);
            else
                for (var u = o && e[0].length, s = 0; s < o; s++)
                    for (var h = 0; h < u; h++) n[s][h] = a(e[s][h], t[s][h], r)
        }

        function u(e, t, r) {
            var n = e.length,
                i = t.length;
            if (n !== i) {
                if (n > i) e.length = i;
                else
                    for (var a = n; a < i; a++) e.push(1 === r ? t[a] : x.call(t[a]))
            }
            for (var o = e[0] && e[0].length, a = 0; a < e.length; a++)
                if (1 === r) isNaN(e[a]) && (e[a] = t[a]);
                else
                    for (var s = 0; s < o; s++) isNaN(e[a][s]) && (e[a][s] = t[a][s])
        }

        function h(e, t, r) {
            if (e === t) return !0;
            var n = e.length;
            if (n !== t.length) return !1;
            if (1 === r) {
                for (var i = 0; i < n; i++)
                    if (e[i] !== t[i]) return !1
            } else
                for (var a = e[0].length, i = 0; i < n; i++)
                    for (var o = 0; o < a; o++)
                        if (e[i][o] !== t[i][o]) return !1; return !0
        }

        function l(e, t, r, n, i, a, o, s, u) {
            var h = e.length;
            if (1 == u)
                for (var l = 0; l < h; l++) s[l] = c(e[l], t[l], r[l], n[l], i, a, o);
            else
                for (var d = e[0].length, l = 0; l < h; l++)
                    for (var f = 0; f < d; f++) s[l][f] = c(e[l][f], t[l][f], r[l][f], n[l][f], i, a, o)
        }

        function c(e, t, r, n, i, a, o) {
            var s = .5 * (r - e),
                u = .5 * (n - t);
            return (2 * (t - r) + s + u) * o + (-3 * (t - r) - 2 * s - u) * a + s * i + t
        }

        function d(e) { if (y(e)) { var t = e.length; if (y(e[0])) { for (var r = [], n = 0; n < t; n++) r.push(x.call(e[n])); return r } return x.call(e) } return e }

        function f(e) { return e[0] = Math.floor(e[0]), e[1] = Math.floor(e[1]), e[2] = Math.floor(e[2]), "rgba(" + e.join(",") + ")" }

        function p(e) { var t = e[e.length - 1].value; return y(t && t[0]) ? 2 : 1 }

        function _(e, t, r, n, i, d) {
            var _ = e._getter,
                v = e._setter,
                x = "spline" === t,
                T = n.length;
            if (T) {
                var b, w = n[0].value,
                    E = y(w),
                    S = !1,
                    A = !1,
                    M = E ? p(n) : 0;
                n.sort(function(e, t) { return e.time - t.time }), b = n[T - 1].time;
                for (var N = [], C = [], L = n[0].value, D = !0, I = 0; I < T; I++) {
                    N.push(n[I].time / b);
                    var R = n[I].value;
                    if (E && h(R, L, M) || !E && R === L || (D = !1), L = R, "string" == typeof R) {
                        var P = g.parse(R);
                        P ? (R = P, S = !0) : A = !0
                    }
                    C.push(R)
                }
                if (d || !D) {
                    for (var O = C[T - 1], I = 0; I < T - 1; I++) E ? u(C[I], O, M) : !isNaN(C[I]) || isNaN(O) || A || S || (C[I] = O);
                    E && u(_(e._target, i), O, M);
                    var F, B, U, z, G, k, H = 0,
                        V = 0;
                    if (S) var W = [0, 0, 0, 0];
                    var q = function(e, t) {
                            var r;
                            if (t < 0) r = 0;
                            else if (t < V) {
                                for (F = Math.min(H + 1, T - 1), r = F; r >= 0 && !(N[r] <= t); r--);
                                r = Math.min(r, T - 2)
                            } else {
                                for (r = H; r < T && !(N[r] > t); r++);
                                r = Math.min(r - 1, T - 2)
                            }
                            H = r, V = t;
                            var n = N[r + 1] - N[r];
                            if (0 !== n)
                                if (B = (t - N[r]) / n, x)
                                    if (z = C[r], U = C[0 === r ? r : r - 1], G = C[r > T - 2 ? T - 1 : r + 1], k = C[r > T - 3 ? T - 1 : r + 2], E) l(U, z, G, k, B, B * B, B * B * B, _(e, i), M);
                                    else {
                                        var u;
                                        if (S) u = l(U, z, G, k, B, B * B, B * B * B, W, 1), u = f(W);
                                        else {
                                            if (A) return o(z, G, B);
                                            u = c(U, z, G, k, B, B * B, B * B * B)
                                        }
                                        v(e, i, u)
                                    }
                            else if (E) s(C[r], C[r + 1], B, _(e, i), M);
                            else {
                                var u;
                                if (S) s(C[r], C[r + 1], B, W, 1), u = f(W);
                                else {
                                    if (A) return o(C[r], C[r + 1], B);
                                    u = a(C[r], C[r + 1], B)
                                }
                                v(e, i, u)
                            }
                        },
                        X = new m({ target: e._target, life: b, loop: e._loop, delay: e._delay, onframe: q, ondestroy: r });
                    return t && "spline" !== t && (X.easing = t), X
                }
            }
        }
        var m = r(237),
            g = r(244),
            v = r(15),
            y = v.isArrayLike,
            x = Array.prototype.slice,
            T = function(e, t, r, a) { this._tracks = {}, this._target = e, this._loop = t || !1, this._getter = r || n, this._setter = a || i, this._clipCount = 0, this._delay = 0, this._doneList = [], this._onframeList = [], this._clipList = [] };
        T.prototype = {
            when: function(e, t) {
                var r = this._tracks;
                for (var n in t)
                    if (t.hasOwnProperty(n)) {
                        if (!r[n]) {
                            r[n] = [];
                            var i = this._getter(this._target, n);
                            if (null == i) continue;
                            0 !== e && r[n].push({ time: 0, value: d(i) })
                        }
                        r[n].push({ time: e, value: t[n] })
                    }
                return this
            },
            during: function(e) { return this._onframeList.push(e), this },
            pause: function() {
                for (var e = 0; e < this._clipList.length; e++) this._clipList[e].pause();
                this._paused = !0
            },
            resume: function() {
                for (var e = 0; e < this._clipList.length; e++) this._clipList[e].resume();
                this._paused = !1
            },
            isPaused: function() { return !!this._paused },
            _doneCallback: function() { this._tracks = {}, this._clipList.length = 0; for (var e = this._doneList, t = e.length, r = 0; r < t; r++) e[r].call(this) },
            start: function(e, t) {
                var r, n = this,
                    i = 0,
                    a = function() {--i || n._doneCallback() };
                for (var o in this._tracks)
                    if (this._tracks.hasOwnProperty(o)) {
                        var s = _(this, e, a, this._tracks[o], o, t);
                        s && (this._clipList.push(s), i++, this.animation && this.animation.addClip(s), r = s)
                    }
                if (r) {
                    var u = r.onframe;
                    r.onframe = function(e, t) { u(e, t); for (var r = 0; r < n._onframeList.length; r++) n._onframeList[r](e, t) }
                }
                return i || this._doneCallback(), this
            },
            stop: function(e) {
                for (var t = this._clipList, r = this.animation, n = 0; n < t.length; n++) {
                    var i = t[n];
                    e && i.onframe(this._target, 1), r && r.removeClip(i)
                }
                t.length = 0
            },
            delay: function(e) { return this._delay = e, this },
            done: function(e) { return e && this._doneList.push(e), this },
            getClips: function() { return this._clipList }
        }, e.exports = T
    }, function(e, t, r) {
        function n(e) { this._target = e.target, this._life = e.life || 1e3, this._delay = e.delay || 0, this._initialized = !1, this.loop = null != e.loop && e.loop, this.gap = e.gap || 0, this.easing = e.easing || "Linear", this.onframe = e.onframe, this.ondestroy = e.ondestroy, this.onrestart = e.onrestart, this._pausedTime = 0, this._paused = !1 }
        var i = r(238);
        n.prototype = {
            constructor: n,
            step: function(e, t) {
                if (this._initialized || (this._startTime = e + this._delay, this._initialized = !0), this._paused) return void(this._pausedTime += t);
                var r = (e - this._startTime - this._pausedTime) / this._life;
                if (!(r < 0)) {
                    r = Math.min(r, 1);
                    var n = this.easing,
                        a = "string" == typeof n ? i[n] : n,
                        o = "function" == typeof a ? a(r) : r;
                    return this.fire("frame", o), 1 == r ? this.loop ? (this.restart(e), "restart") : (this._needsRemove = !0, "destroy") : null
                }
            },
            restart: function(e) {
                var t = (e - this._startTime - this._pausedTime) % this._life;
                this._startTime = e - t + this.gap, this._pausedTime = 0, this._needsRemove = !1
            },
            fire: function(e, t) { e = "on" + e, this[e] && this[e](this._target, t) },
            pause: function() { this._paused = !0 },
            resume: function() { this._paused = !1 }
        }, e.exports = n
    }, function(e, t) {
        var r = { linear: function(e) { return e }, quadraticIn: function(e) { return e * e }, quadraticOut: function(e) { return e * (2 - e) }, quadraticInOut: function(e) { return (e *= 2) < 1 ? .5 * e * e : -.5 * (--e * (e - 2) - 1) }, cubicIn: function(e) { return e * e * e }, cubicOut: function(e) { return --e * e * e + 1 }, cubicInOut: function(e) { return (e *= 2) < 1 ? .5 * e * e * e : .5 * ((e -= 2) * e * e + 2) }, quarticIn: function(e) { return e * e * e * e }, quarticOut: function(e) { return 1 - --e * e * e * e }, quarticInOut: function(e) { return (e *= 2) < 1 ? .5 * e * e * e * e : -.5 * ((e -= 2) * e * e * e - 2) }, quinticIn: function(e) { return e * e * e * e * e }, quinticOut: function(e) { return --e * e * e * e * e + 1 }, quinticInOut: function(e) { return (e *= 2) < 1 ? .5 * e * e * e * e * e : .5 * ((e -= 2) * e * e * e * e + 2) }, sinusoidalIn: function(e) { return 1 - Math.cos(e * Math.PI / 2) }, sinusoidalOut: function(e) { return Math.sin(e * Math.PI / 2) }, sinusoidalInOut: function(e) { return .5 * (1 - Math.cos(Math.PI * e)) }, exponentialIn: function(e) { return 0 === e ? 0 : Math.pow(1024, e - 1) }, exponentialOut: function(e) { return 1 === e ? 1 : 1 - Math.pow(2, -10 * e) }, exponentialInOut: function(e) { return 0 === e ? 0 : 1 === e ? 1 : (e *= 2) < 1 ? .5 * Math.pow(1024, e - 1) : .5 * (2 - Math.pow(2, -10 * (e - 1))) }, circularIn: function(e) { return 1 - Math.sqrt(1 - e * e) }, circularOut: function(e) { return Math.sqrt(1 - --e * e) }, circularInOut: function(e) { return (e *= 2) < 1 ? -.5 * (Math.sqrt(1 - e * e) - 1) : .5 * (Math.sqrt(1 - (e -= 2) * e) + 1) }, elasticIn: function(e) { var t, r = .1; return 0 === e ? 0 : 1 === e ? 1 : (!r || r < 1 ? (r = 1, t = .1) : t = .4 * Math.asin(1 / r) / (2 * Math.PI), -r * Math.pow(2, 10 * (e -= 1)) * Math.sin((e - t) * (2 * Math.PI) / .4)) }, elasticOut: function(e) { var t, r = .1; return 0 === e ? 0 : 1 === e ? 1 : (!r || r < 1 ? (r = 1, t = .1) : t = .4 * Math.asin(1 / r) / (2 * Math.PI), r * Math.pow(2, -10 * e) * Math.sin((e - t) * (2 * Math.PI) / .4) + 1) }, elasticInOut: function(e) { var t, r = .1; return 0 === e ? 0 : 1 === e ? 1 : (!r || r < 1 ? (r = 1, t = .1) : t = .4 * Math.asin(1 / r) / (2 * Math.PI), (e *= 2) < 1 ? r * Math.pow(2, 10 * (e -= 1)) * Math.sin((e - t) * (2 * Math.PI) / .4) * -.5 : r * Math.pow(2, -10 * (e -= 1)) * Math.sin((e - t) * (2 * Math.PI) / .4) * .5 + 1) }, backIn: function(e) { var t = 1.70158; return e * e * ((t + 1) * e - t) }, backOut: function(e) { var t = 1.70158; return --e * e * ((t + 1) * e + t) + 1 }, backInOut: function(e) { var t = 2.5949095; return (e *= 2) < 1 ? e * e * ((t + 1) * e - t) * .5 : .5 * ((e -= 2) * e * ((t + 1) * e + t) + 2) }, bounceIn: function(e) { return 1 - r.bounceOut(1 - e) }, bounceOut: function(e) { return e < 1 / 2.75 ? 7.5625 * e * e : e < 2 / 2.75 ? 7.5625 * (e -= 1.5 / 2.75) * e + .75 : e < 2.5 / 2.75 ? 7.5625 * (e -= 2.25 / 2.75) * e + .9375 : 7.5625 * (e -= 2.625 / 2.75) * e + .984375 }, bounceInOut: function(e) { return e < .5 ? .5 * r.bounceIn(2 * e) : .5 * r.bounceOut(2 * e - 1) + .5 } };
        e.exports = r
    }, function(e, t) {
        e.exports = {
            containStroke: function(e, t, r, n, i, a, o) {
                if (0 === i) return !1;
                var s = i,
                    u = 0,
                    h = e;
                if (o > t + s && o > n + s || o < t - s && o < n - s || a > e + s && a > r + s || a < e - s && a < r - s) return !1;
                if (e === r) return Math.abs(a - e) <= s / 2;
                u = (t - n) / (e - r), h = (e * n - r * t) / (e - r);
                var l = u * a - o + h;
                return l * l / (u * u + 1) <= s / 2 * s / 2
            }
        }
    }, function(e, t, r) {
        function n(e, t) { t = t || M; var r = e + ":" + t; if (w[r]) return w[r]; for (var n = (e + "").split("\n"), i = 0, a = 0, o = n.length; a < o; a++) i = Math.max(L.measureText(n[a], t).width, i); return E > S && (E = 0, w = {}), E++, w[r] = i, i }

        function i(e, t, r, n, i, s, u) { return s ? o(e, t, r, n, i, s, u) : a(e, t, r, n, i, u) }

        function a(e, t, r, i, a, o) {
            var h = m(e, t, a, o),
                l = n(e, t);
            a && (l += a[1] + a[3]);
            var c = h.outerHeight,
                d = s(0, l, r),
                f = u(0, c, i),
                p = new T(d, f, l, c);
            return p.lineHeight = h.lineHeight, p
        }

        function o(e, t, r, n, i, a, o) {
            var h = g(e, { rich: a, truncate: o, font: t, textAlign: r, textPadding: i }),
                l = h.outerWidth,
                c = h.outerHeight,
                d = s(0, l, r),
                f = u(0, c, n);
            return new T(d, f, l, c)
        }

        function s(e, t, r) { return "right" === r ? e -= t : "center" === r && (e -= t / 2), e }

        function u(e, t, r) { return "middle" === r ? e -= t / 2 : "bottom" === r && (e -= t), e }

        function h(e, t, r) {
            var n = t.x,
                i = t.y,
                a = t.height,
                o = t.width,
                s = a / 2,
                u = "left",
                h = "top";
            switch (e) {
                case "left":
                    n -= r, i += s, u = "right", h = "middle";
                    break;
                case "right":
                    n += r + o, i += s, h = "middle";
                    break;
                case "top":
                    n += o / 2, i -= r, u = "center", h = "bottom";
                    break;
                case "bottom":
                    n += o / 2, i += a + r, u = "center";
                    break;
                case "inside":
                    n += o / 2, i += s, u = "center", h = "middle";
                    break;
                case "insideLeft":
                    n += r, i += s, h = "middle";
                    break;
                case "insideRight":
                    n += o - r, i += s, u = "right", h = "middle";
                    break;
                case "insideTop":
                    n += o / 2, i += r, u = "center";
                    break;
                case "insideBottom":
                    n += o / 2, i += a - r, u = "center", h = "bottom";
                    break;
                case "insideTopLeft":
                    n += r, i += r;
                    break;
                case "insideTopRight":
                    n += o - r, i += r, u = "right";
                    break;
                case "insideBottomLeft":
                    n += r, i += a - r, h = "bottom";
                    break;
                case "insideBottomRight":
                    n += o - r, i += a - r, u = "right", h = "bottom"
            }
            return { x: n, y: i, textAlign: u, textVerticalAlign: h }
        }

        function l(e, t, r, n, i) {
            if (!t) return "";
            var a = (e + "").split("\n");
            i = c(t, r, n, i);
            for (var o = 0, s = a.length; o < s; o++) a[o] = d(a[o], i);
            return a.join("\n")
        }

        function c(e, t, r, i) {
            i = x.extend({}, i), i.font = t;
            var r = N(r, "...");
            i.maxIterations = N(i.maxIterations, 2);
            var a = i.minChar = N(i.minChar, 0);
            i.cnCharWidth = n("å›½", t);
            var o = i.ascCharWidth = n("a", t);
            i.placeholder = N(i.placeholder, "");
            for (var s = e = Math.max(0, e - 1), u = 0; u < a && s >= o; u++) s -= o;
            var h = n(r);
            return h > s && (r = "", h = 0), s = e - h, i.ellipsis = r, i.ellipsisWidth = h, i.contentWidth = s, i.containerWidth = e, i
        }

        function d(e, t) {
            var r = t.containerWidth,
                i = t.font,
                a = t.contentWidth;
            if (!r) return "";
            var o = n(e, i);
            if (o <= r) return e;
            for (var s = 0;; s++) {
                if (o <= a || s >= t.maxIterations) { e += t.ellipsis; break }
                var u = 0 === s ? f(e, a, t.ascCharWidth, t.cnCharWidth) : o > 0 ? Math.floor(e.length * a / o) : 0;
                e = e.substr(0, u), o = n(e, i)
            }
            return "" === e && (e = t.placeholder), e
        }

        function f(e, t, r, n) {
            for (var i = 0, a = 0, o = e.length; a < o && i < t; a++) {
                var s = e.charCodeAt(a);
                i += 0 <= s && s <= 127 ? r : n
            }
            return a
        }

        function p(e) { return n("å›½", e) }

        function _(e, t) { var r = x.getContext(); return r.font = t || M, r.measureText(e) }

        function m(e, t, r, n) {
            null != e && (e += "");
            var i = p(t),
                a = e ? e.split("\n") : [],
                o = a.length * i,
                s = o;
            if (r && (s += r[0] + r[2]), e && n) {
                var u = n.outerHeight,
                    h = n.outerWidth;
                if (null != u && s > u) e = "", a = [];
                else if (null != h)
                    for (var l = c(h - (r ? r[1] + r[3] : 0), t, n.ellipsis, { minChar: n.minChar, placeholder: n.placeholder }), f = 0, _ = a.length; f < _; f++) a[f] = d(a[f], l)
            }
            return { lines: a, height: o, outerHeight: s, lineHeight: i }
        }

        function g(e, t) {
            var r = { lines: [], width: 0, height: 0 };
            if (null != e && (e += ""), !e) return r;
            for (var n, i = A.lastIndex = 0; null != (n = A.exec(e));) {
                var a = n.index;
                a > i && v(r, e.substring(i, a)), v(r, n[2], n[1]), i = A.lastIndex
            }
            i < e.length && v(r, e.substring(i, e.length));
            var o = r.lines,
                s = 0,
                u = 0,
                h = [],
                c = t.textPadding,
                d = t.truncate,
                f = d && d.outerWidth,
                p = d && d.outerHeight;
            c && (null != f && (f -= c[1] + c[3]), null != p && (p -= c[0] + c[2]));
            for (var _ = 0; _ < o.length; _++) {
                for (var m = o[_], g = 0, y = 0, x = 0; x < m.tokens.length; x++) {
                    var T = m.tokens[x],
                        w = T.styleName && t.rich[T.styleName] || {},
                        E = T.textPadding = w.textPadding,
                        S = T.font = w.font || t.font,
                        M = T.textHeight = N(w.textHeight, L.getLineHeight(S));
                    if (E && (M += E[0] + E[2]), T.height = M, T.lineHeight = C(w.textLineHeight, t.textLineHeight, M), T.textAlign = w && w.textAlign || t.textAlign, T.textVerticalAlign = w && w.textVerticalAlign || "middle", null != p && s + T.lineHeight > p) return { lines: [], width: 0, height: 0 };
                    T.textWidth = L.getWidth(T.text, S);
                    var D = w.textWidth,
                        I = null == D || "auto" === D;
                    if ("string" == typeof D && "%" === D.charAt(D.length - 1)) T.percentWidth = D, h.push(T), D = 0;
                    else {
                        if (I) {
                            D = T.textWidth;
                            var R = w.textBackgroundColor,
                                P = R && R.image;
                            P && (P = b.findExistImage(P), b.isImageReady(P) && (D = Math.max(D, P.width * M / P.height)))
                        }
                        var O = E ? E[1] + E[3] : 0;
                        D += O;
                        var F = null != f ? f - y : null;
                        null != F && F < D && (!I || F < O ? (T.text = "", T.textWidth = D = 0) : (T.text = l(T.text, F - O, S, d.ellipsis, { minChar: d.minChar }), T.textWidth = L.getWidth(T.text, S), D = T.textWidth + O))
                    }
                    y += T.width = D, w && (g = Math.max(g, T.lineHeight))
                }
                m.width = y, m.lineHeight = g, s += g, u = Math.max(u, y)
            }
            r.outerWidth = r.width = N(t.textWidth, u), r.outerHeight = r.height = N(t.textHeight, s), c && (r.outerWidth += c[1] + c[3], r.outerHeight += c[0] + c[2]);
            for (var _ = 0; _ < h.length; _++) {
                var T = h[_],
                    B = T.percentWidth;
                T.width = parseInt(B, 10) / 100 * u
            }
            return r
        }

        function v(e, t, r) {
            for (var n = "" === t, i = t.split("\n"), a = e.lines, o = 0; o < i.length; o++) {
                var s = i[o],
                    u = { styleName: r, text: s, isLineHolder: !s && !n };
                if (o) a.push({ tokens: [u] });
                else {
                    var h = (a[a.length - 1] || (a[0] = { tokens: [] })).tokens,
                        l = h.length;
                    1 === l && h[0].isLineHolder ? h[0] = u : (s || !l || n) && h.push(u)
                }
            }
        }

        function y(e) { return (e.fontSize || e.fontFamily) && [e.fontStyle, e.fontWeight, (e.fontSize || 12) + "px", e.fontFamily || "sans-serif"].join(" ") || e.textFont || e.font }
        var x = r(15),
            T = r(84),
            b = r(243),
            w = {},
            E = 0,
            S = 5e3,
            A = /\{([a-zA-Z0-9_]+)\|([^}]*)\}/g,
            M = "12px sans-serif",
            N = x.retrieve2,
            C = x.retrieve3,
            L = { getWidth: n, getBoundingRect: i, adjustTextPositionOnRect: h, truncateText: l, measureText: _, getLineHeight: p, parsePlainText: m, parseRichText: g, adjustTextX: s, adjustTextY: u, makeFont: y, DEFAULT_FONT: M };
        e.exports = L
    }, function(e, t) {
        var r = "undefined" == typeof Float32Array ? Array : Float32Array,
            n = {
                create: function() { var e = new r(6); return n.identity(e), e },
                identity: function(e) { return e[0] = 1, e[1] = 0, e[2] = 0, e[3] = 1, e[4] = 0, e[5] = 0, e },
                copy: function(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4], e[5] = t[5], e },
                mul: function(e, t, r) {
                    var n = t[0] * r[0] + t[2] * r[1],
                        i = t[1] * r[0] + t[3] * r[1],
                        a = t[0] * r[2] + t[2] * r[3],
                        o = t[1] * r[2] + t[3] * r[3],
                        s = t[0] * r[4] + t[2] * r[5] + t[4],
                        u = t[1] * r[4] + t[3] * r[5] + t[5];
                    return e[0] = n, e[1] = i, e[2] = a, e[3] = o, e[4] = s, e[5] = u, e
                },
                translate: function(e, t, r) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e[4] = t[4] + r[0], e[5] = t[5] + r[1], e },
                rotate: function(e, t, r) {
                    var n = t[0],
                        i = t[2],
                        a = t[4],
                        o = t[1],
                        s = t[3],
                        u = t[5],
                        h = Math.sin(r),
                        l = Math.cos(r);
                    return e[0] = n * l + o * h, e[1] = -n * h + o * l, e[2] = i * l + s * h, e[3] = -i * h + l * s, e[4] = l * a + h * u, e[5] = l * u - h * a, e
                },
                scale: function(e, t, r) {
                    var n = r[0],
                        i = r[1];
                    return e[0] = t[0] * n, e[1] = t[1] * i, e[2] = t[2] * n, e[3] = t[3] * i, e[4] = t[4] * n, e[5] = t[5] * i, e
                },
                invert: function(e, t) {
                    var r = t[0],
                        n = t[2],
                        i = t[4],
                        a = t[1],
                        o = t[3],
                        s = t[5],
                        u = r * o - a * n;
                    return u ? (u = 1 / u, e[0] = o * u, e[1] = -a * u, e[2] = -n * u, e[3] = r * u, e[4] = (n * s - o * i) * u, e[5] = (a * i - r * s) * u, e) : null
                }
            };
        e.exports = n
    }, function(e, t) {
        var r = "undefined" == typeof Float32Array ? Array : Float32Array,
            n = {
                create: function(e, t) { var n = new r(2); return null == e && (e = 0), null == t && (t = 0), n[0] = e, n[1] = t, n },
                copy: function(e, t) { return e[0] = t[0], e[1] = t[1], e },
                clone: function(e) { var t = new r(2); return t[0] = e[0], t[1] = e[1], t },
                set: function(e, t, r) { return e[0] = t, e[1] = r, e },
                add: function(e, t, r) { return e[0] = t[0] + r[0], e[1] = t[1] + r[1], e },
                scaleAndAdd: function(e, t, r, n) { return e[0] = t[0] + r[0] * n, e[1] = t[1] + r[1] * n, e },
                sub: function(e, t, r) { return e[0] = t[0] - r[0], e[1] = t[1] - r[1], e },
                len: function(e) { return Math.sqrt(this.lenSquare(e)) },
                lenSquare: function(e) { return e[0] * e[0] + e[1] * e[1] },
                mul: function(e, t, r) { return e[0] = t[0] * r[0], e[1] = t[1] * r[1], e },
                div: function(e, t, r) { return e[0] = t[0] / r[0], e[1] = t[1] / r[1], e },
                dot: function(e, t) { return e[0] * t[0] + e[1] * t[1] },
                scale: function(e, t, r) { return e[0] = t[0] * r, e[1] = t[1] * r, e },
                normalize: function(e, t) { var r = n.len(t); return 0 === r ? (e[0] = 0, e[1] = 0) : (e[0] = t[0] / r, e[1] = t[1] / r), e },
                distance: function(e, t) { return Math.sqrt((e[0] - t[0]) * (e[0] - t[0]) + (e[1] - t[1]) * (e[1] - t[1])) },
                distanceSquare: function(e, t) { return (e[0] - t[0]) * (e[0] - t[0]) + (e[1] - t[1]) * (e[1] - t[1]) },
                negate: function(e, t) { return e[0] = -t[0], e[1] = -t[1], e },
                lerp: function(e, t, r, n) { return e[0] = t[0] + n * (r[0] - t[0]), e[1] = t[1] + n * (r[1] - t[1]), e },
                applyTransform: function(e, t, r) {
                    var n = t[0],
                        i = t[1];
                    return e[0] = r[0] * n + r[2] * i + r[4], e[1] = r[1] * n + r[3] * i + r[5], e
                },
                min: function(e, t, r) { return e[0] = Math.min(t[0], r[0]), e[1] = Math.min(t[1], r[1]), e },
                max: function(e, t, r) { return e[0] = Math.max(t[0], r[0]), e[1] = Math.max(t[1], r[1]), e }
            };
        n.length = n.len, n.lengthSquare = n.lenSquare, n.dist = n.distance, n.distSquare = n.distanceSquare, e.exports = n
    }, function(e, t, r) {
        function n() {
            var e = this.__cachedImgObj;
            this.onload = this.__cachedImgObj = null;
            for (var t = 0; t < e.pending.length; t++) {
                var r = e.pending[t],
                    n = r.cb;
                n && n(this, r.cbPayload), r.hostEl.dirty()
            }
            e.pending.length = 0
        }
        var i = r(60),
            a = new i(50),
            o = {};
        o.findExistImage = function(e) { if ("string" == typeof e) { var t = a.get(e); return t && t.image } return e }, o.createOrUpdateImage = function(e, t, r, i, o) {
            if (e) {
                if ("string" == typeof e) {
                    if (t && t.__zrImageSrc === e || !r) return t;
                    var u = a.get(e),
                        h = { hostEl: r, cb: i, cbPayload: o };
                    return u ? (t = u.image, !s(t) && u.pending.push(h)) : (!t && (t = new Image), t.onload = n, a.put(e, t.__cachedImgObj = { image: t, pending: [h] }), t.src = t.__zrImageSrc = e), t
                }
                return e
            }
            return t
        };
        var s = o.isImageReady = function(e) { return e && e.width && e.height };
        e.exports = o
    }, function(e, t, r) {
        function n(e) { return e = Math.round(e), e < 0 ? 0 : e > 255 ? 255 : e }

        function i(e) { return e = Math.round(e), e < 0 ? 0 : e > 360 ? 360 : e }

        function a(e) { return e < 0 ? 0 : e > 1 ? 1 : e }

        function o(e) { return n(e.length && "%" === e.charAt(e.length - 1) ? parseFloat(e) / 100 * 255 : parseInt(e, 10)) }

        function s(e) { return a(e.length && "%" === e.charAt(e.length - 1) ? parseFloat(e) / 100 : parseFloat(e)) }

        function u(e, t, r) { return r < 0 ? r += 1 : r > 1 && (r -= 1), 6 * r < 1 ? e + (t - e) * r * 6 : 2 * r < 1 ? t : 3 * r < 2 ? e + (t - e) * (2 / 3 - r) * 6 : e }

        function h(e, t, r) { return e + (t - e) * r }

        function l(e, t, r, n, i) { return e[0] = t, e[1] = r, e[2] = n, e[3] = i, e }

        function c(e, t) { return e[0] = t[0], e[1] = t[1], e[2] = t[2], e[3] = t[3], e }

        function d(e, t) { A && c(A, t), A = S.put(e, A || t.slice()) }

        function f(e, t) {
            if (e) {
                t = t || [];
                var r = S.get(e);
                if (r) return c(t, r);
                e += "";
                var n = e.replace(/ /g, "").toLowerCase();
                if (n in E) return c(t, E[n]), d(e, t), t;
                if ("#" !== n.charAt(0)) {
                    var i = n.indexOf("("),
                        a = n.indexOf(")");
                    if (-1 !== i && a + 1 === n.length) {
                        var u = n.substr(0, i),
                            h = n.substr(i + 1, a - (i + 1)).split(","),
                            f = 1;
                        switch (u) {
                            case "rgba":
                                if (4 !== h.length) return void l(t, 0, 0, 0, 1);
                                f = s(h.pop());
                            case "rgb":
                                return 3 !== h.length ? void l(t, 0, 0, 0, 1) : (l(t, o(h[0]), o(h[1]), o(h[2]), f), d(e, t), t);
                            case "hsla":
                                return 4 !== h.length ? void l(t, 0, 0, 0, 1) : (h[3] = s(h[3]), p(h, t), d(e, t), t);
                            case "hsl":
                                return 3 !== h.length ? void l(t, 0, 0, 0, 1) : (p(h, t), d(e, t), t);
                            default:
                                return
                        }
                    }
                    l(t, 0, 0, 0, 1)
                } else { if (4 === n.length) { var _ = parseInt(n.substr(1), 16); return _ >= 0 && _ <= 4095 ? (l(t, (3840 & _) >> 4 | (3840 & _) >> 8, 240 & _ | (240 & _) >> 4, 15 & _ | (15 & _) << 4, 1), d(e, t), t) : void l(t, 0, 0, 0, 1) } if (7 === n.length) { var _ = parseInt(n.substr(1), 16); return _ >= 0 && _ <= 16777215 ? (l(t, (16711680 & _) >> 16, (65280 & _) >> 8, 255 & _, 1), d(e, t), t) : void l(t, 0, 0, 0, 1) } }
            }
        }

        function p(e, t) {
            var r = (parseFloat(e[0]) % 360 + 360) % 360 / 360,
                i = s(e[1]),
                a = s(e[2]),
                o = a <= .5 ? a * (i + 1) : a + i - a * i,
                h = 2 * a - o;
            return t = t || [], l(t, n(255 * u(h, o, r + 1 / 3)), n(255 * u(h, o, r)), n(255 * u(h, o, r - 1 / 3)), 1), 4 === e.length && (t[3] = e[3]), t
        }

        function _(e) {
            if (e) {
                var t, r, n = e[0] / 255,
                    i = e[1] / 255,
                    a = e[2] / 255,
                    o = Math.min(n, i, a),
                    s = Math.max(n, i, a),
                    u = s - o,
                    h = (s + o) / 2;
                if (0 === u) t = 0, r = 0;
                else {
                    r = h < .5 ? u / (s + o) : u / (2 - s - o);
                    var l = ((s - n) / 6 + u / 2) / u,
                        c = ((s - i) / 6 + u / 2) / u,
                        d = ((s - a) / 6 + u / 2) / u;
                    n === s ? t = d - c : i === s ? t = 1 / 3 + l - d : a === s && (t = 2 / 3 + c - l), t < 0 && (t += 1), t > 1 && (t -= 1)
                }
                var f = [360 * t, r, h];
                return null != e[3] && f.push(e[3]), f
            }
        }

        function m(e, t) { var r = f(e); if (r) { for (var n = 0; n < 3; n++) r[n] = t < 0 ? r[n] * (1 - t) | 0 : (255 - r[n]) * t + r[n] | 0; return b(r, 4 === r.length ? "rgba" : "rgb") } }

        function g(e, t) { var r = f(e); if (r) return ((1 << 24) + (r[0] << 16) + (r[1] << 8) + +r[2]).toString(16).slice(1) }

        function v(e, t, r) {
            if (t && t.length && e >= 0 && e <= 1) {
                r = r || [];
                var i = e * (t.length - 1),
                    o = Math.floor(i),
                    s = Math.ceil(i),
                    u = t[o],
                    l = t[s],
                    c = i - o;
                return r[0] = n(h(u[0], l[0], c)), r[1] = n(h(u[1], l[1], c)), r[2] = n(h(u[2], l[2], c)), r[3] = a(h(u[3], l[3], c)), r
            }
        }

        function y(e, t, r) {
            if (t && t.length && e >= 0 && e <= 1) {
                var i = e * (t.length - 1),
                    o = Math.floor(i),
                    s = Math.ceil(i),
                    u = f(t[o]),
                    l = f(t[s]),
                    c = i - o,
                    d = b([n(h(u[0], l[0], c)), n(h(u[1], l[1], c)), n(h(u[2], l[2], c)), a(h(u[3], l[3], c))], "rgba");
                return r ? { color: d, leftIndex: o, rightIndex: s, value: i } : d
            }
        }

        function x(e, t, r, n) { if (e = f(e)) return e = _(e), null != t && (e[0] = i(t)), null != r && (e[1] = s(r)), null != n && (e[2] = s(n)), b(p(e), "rgba") }

        function T(e, t) { if ((e = f(e)) && null != t) return e[3] = a(t), b(e, "rgba") }

        function b(e, t) { if (e && e.length) { var r = e[0] + "," + e[1] + "," + e[2]; return "rgba" !== t && "hsva" !== t && "hsla" !== t || (r += "," + e[3]), t + "(" + r + ")" } }
        var w = r(60),
            E = { transparent: [0, 0, 0, 0], aliceblue: [240, 248, 255, 1], antiquewhite: [250, 235, 215, 1], aqua: [0, 255, 255, 1], aquamarine: [127, 255, 212, 1], azure: [240, 255, 255, 1], beige: [245, 245, 220, 1], bisque: [255, 228, 196, 1], black: [0, 0, 0, 1], blanchedalmond: [255, 235, 205, 1], blue: [0, 0, 255, 1], blueviolet: [138, 43, 226, 1], brown: [165, 42, 42, 1], burlywood: [222, 184, 135, 1], cadetblue: [95, 158, 160, 1], chartreuse: [127, 255, 0, 1], chocolate: [210, 105, 30, 1], coral: [255, 127, 80, 1], cornflowerblue: [100, 149, 237, 1], cornsilk: [255, 248, 220, 1], crimson: [220, 20, 60, 1], cyan: [0, 255, 255, 1], darkblue: [0, 0, 139, 1], darkcyan: [0, 139, 139, 1], darkgoldenrod: [184, 134, 11, 1], darkgray: [169, 169, 169, 1], darkgreen: [0, 100, 0, 1], darkgrey: [169, 169, 169, 1], darkkhaki: [189, 183, 107, 1], darkmagenta: [139, 0, 139, 1], darkolivegreen: [85, 107, 47, 1], darkorange: [255, 140, 0, 1], darkorchid: [153, 50, 204, 1], darkred: [139, 0, 0, 1], darksalmon: [233, 150, 122, 1], darkseagreen: [143, 188, 143, 1], darkslateblue: [72, 61, 139, 1], darkslategray: [47, 79, 79, 1], darkslategrey: [47, 79, 79, 1], darkturquoise: [0, 206, 209, 1], darkviolet: [148, 0, 211, 1], deeppink: [255, 20, 147, 1], deepskyblue: [0, 191, 255, 1], dimgray: [105, 105, 105, 1], dimgrey: [105, 105, 105, 1], dodgerblue: [30, 144, 255, 1], firebrick: [178, 34, 34, 1], floralwhite: [255, 250, 240, 1], forestgreen: [34, 139, 34, 1], fuchsia: [255, 0, 255, 1], gainsboro: [220, 220, 220, 1], ghostwhite: [248, 248, 255, 1], gold: [255, 215, 0, 1], goldenrod: [218, 165, 32, 1], gray: [128, 128, 128, 1], green: [0, 128, 0, 1], greenyellow: [173, 255, 47, 1], grey: [128, 128, 128, 1], honeydew: [240, 255, 240, 1], hotpink: [255, 105, 180, 1], indianred: [205, 92, 92, 1], indigo: [75, 0, 130, 1], ivory: [255, 255, 240, 1], khaki: [240, 230, 140, 1], lavender: [230, 230, 250, 1], lavenderblush: [255, 240, 245, 1], lawngreen: [124, 252, 0, 1], lemonchiffon: [255, 250, 205, 1], lightblue: [173, 216, 230, 1], lightcoral: [240, 128, 128, 1], lightcyan: [224, 255, 255, 1], lightgoldenrodyellow: [250, 250, 210, 1], lightgray: [211, 211, 211, 1], lightgreen: [144, 238, 144, 1], lightgrey: [211, 211, 211, 1], lightpink: [255, 182, 193, 1], lightsalmon: [255, 160, 122, 1], lightseagreen: [32, 178, 170, 1], lightskyblue: [135, 206, 250, 1], lightslategray: [119, 136, 153, 1], lightslategrey: [119, 136, 153, 1], lightsteelblue: [176, 196, 222, 1], lightyellow: [255, 255, 224, 1], lime: [0, 255, 0, 1], limegreen: [50, 205, 50, 1], linen: [250, 240, 230, 1], magenta: [255, 0, 255, 1], maroon: [128, 0, 0, 1], mediumaquamarine: [102, 205, 170, 1], mediumblue: [0, 0, 205, 1], mediumorchid: [186, 85, 211, 1], mediumpurple: [147, 112, 219, 1], mediumseagreen: [60, 179, 113, 1], mediumslateblue: [123, 104, 238, 1], mediumspringgreen: [0, 250, 154, 1], mediumturquoise: [72, 209, 204, 1], mediumvioletred: [199, 21, 133, 1], midnightblue: [25, 25, 112, 1], mintcream: [245, 255, 250, 1], mistyrose: [255, 228, 225, 1], moccasin: [255, 228, 181, 1], navajowhite: [255, 222, 173, 1], navy: [0, 0, 128, 1], oldlace: [253, 245, 230, 1], olive: [128, 128, 0, 1], olivedrab: [107, 142, 35, 1], orange: [255, 165, 0, 1], orangered: [255, 69, 0, 1], orchid: [218, 112, 214, 1], palegoldenrod: [238, 232, 170, 1], palegreen: [152, 251, 152, 1], paleturquoise: [175, 238, 238, 1], palevioletred: [219, 112, 147, 1], papayawhip: [255, 239, 213, 1], peachpuff: [255, 218, 185, 1], peru: [205, 133, 63, 1], pink: [255, 192, 203, 1], plum: [221, 160, 221, 1], powderblue: [176, 224, 230, 1], purple: [128, 0, 128, 1], red: [255, 0, 0, 1], rosybrown: [188, 143, 143, 1], royalblue: [65, 105, 225, 1], saddlebrown: [139, 69, 19, 1], salmon: [250, 128, 114, 1], sandybrown: [244, 164, 96, 1], seagreen: [46, 139, 87, 1], seashell: [255, 245, 238, 1], sienna: [160, 82, 45, 1], silver: [192, 192, 192, 1], skyblue: [135, 206, 235, 1], slateblue: [106, 90, 205, 1], slategray: [112, 128, 144, 1], slategrey: [112, 128, 144, 1], snow: [255, 250, 250, 1], springgreen: [0, 255, 127, 1], steelblue: [70, 130, 180, 1], tan: [210, 180, 140, 1], teal: [0, 128, 128, 1], thistle: [216, 191, 216, 1], tomato: [255, 99, 71, 1], turquoise: [64, 224, 208, 1], violet: [238, 130, 238, 1], wheat: [245, 222, 179, 1], white: [255, 255, 255, 1], whitesmoke: [245, 245, 245, 1], yellow: [255, 255, 0, 1], yellowgreen: [154, 205, 50, 1] },
            S = new w(20),
            A = null;
        e.exports = { parse: f, lift: m, toHex: g, fastMapToColor: v, mapToColor: y, modifyHSL: x, modifyAlpha: T, stringify: b }
    }])
});